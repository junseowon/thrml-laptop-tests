import time
import jax
import subprocess

import dwave_networkx
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from thrml.block_management import Block
#from jax.sharding import PartitionSpec as P # <- 삭제 (단일 GPU에 필요 없음)
from thrml.block_sampling import sample_states, SamplingSchedule
from thrml.models.discrete_ebm import SpinEBMFactor
from thrml.models.ising import (
    estimate_kl_grad,
    hinton_init,
    IsingEBM,
    IsingSamplingProgram,
    IsingTrainingSpec,
)
from thrml.pgm import SpinNode

# D-Wave의 그래프 생성 코드를 사용해 그래프 만들기
#graph = dwave_networkx.pegasus_graph(14)

# DWave의 거대한 그래프 대신 20x20 격자 그래프로 축소
graph = nx.grid_graph(dim=(20, 20))

coord_to_node = {coord: SpinNode() for coord in graph.nodes}
nx.relabel_nodes(graph, coord_to_node, copy=False)

nodes = list(graph.nodes)
edges = list(graph.edges)

# 그래프 크기 확인
print(f"그래프 생성 완료. 노드: {len(nodes)}개, 엣지: {len(edges)}개")

seed = 4242
key = jax.random.key(seed)

key, subkey = jax.random.split(key, 2)
biases = jax.random.normal(subkey, (len(nodes),))

key, subkey = jax.random.split(key, 2)
weights = jax.random.normal(subkey, (len(edges),))

beta = jnp.array(1.0)

model = IsingEBM(nodes, edges, biases, weights, beta)

[x.__class__ for x in model.factors]

#n_data = 500  # <- 주석 처리 (400보다 큼)
n_data = 350

np.random.seed(seed)

# 데이터 샘플링 확인
print(f"{len(nodes)}개 노드 중 {n_data}개를 데이터 노드로 샘플링 시도...")
data_inds = np.random.choice(len(graph.nodes), n_data, replace=False)
data_nodes = [nodes[x] for x in data_inds]
print(f"데이터 노드 선택 완료.")

coloring = nx.coloring.greedy_color(graph, strategy="DSATUR")
n_colors = max(coloring.values()) + 1
free_coloring = [[] for _ in range(n_colors)]
# 컬러 그룹 구성
# form color groups
for node in graph.nodes:
    free_coloring[coloring[node]].append(node)

free_blocks = [Block(x) for x in free_coloring]

# 블록(컬러링) 확인
print(f"그래프 컬러링(블록) 완료. 총 {n_colors}개의 블록(색상) 생성됨.")

# 여기서는 동일한 컬러링을 재사용 (항상 가능하지만, 최적은 아님)

# 데이터 노드를 제외한 그래프 복제
graph_copy = graph.copy()
graph_copy.remove_nodes_from(data_nodes)

clamped_coloring = [[] for _ in range(n_colors)]
for node in graph_copy.nodes:
    clamped_coloring[coloring[node]].append(node)

clamped_blocks = [Block(x) for x in clamped_coloring]

# 예시용 임의 데이터 생성
# 실제로는 이미지, 텍스트, 비디오 등으로 대체 가능
data_batch_size = 50

key, subkey = jax.random.split(key, 2)
data = jax.random.bernoulli(subkey, 0.5, (data_batch_size, len(data_nodes))).astype(jnp.bool)

# 학습 및 샘플링에 사용할 스케줄 설정
schedule = SamplingSchedule(5, 100, 5)

# 학습에 필요한 구성요소를 하나로 묶은 객체 생성
training_spec = IsingTrainingSpec(model, [Block(data_nodes)], [], clamped_blocks, free_blocks, schedule, schedule)

# 각 항(term)별로 실행할 병렬 샘플링 체인 개수 설정
n_chains_free = data_batch_size
n_chains_clamped = 1

# 각 샘플링 체인의 초기 상태 생성
# THRML에는 볼츠만 머신에서 자주 사용하는 Hinton 초기화를 위한 코드가 내장되어 있음
key, subkey = jax.random.split(key, 2)
init_state_free = hinton_init(subkey, model, free_blocks, (n_chains_free,))
key, subkey = jax.random.split(key, 2)
init_state_clamped = hinton_init(subkey, model, clamped_blocks, (n_chains_clamped, data_batch_size))

# 그래디언트 계산 (AI 훈련)
print(f"AI 훈련 그래디언트 계산 시작... (시간이 걸릴 수 있습니다)")

# 그래디언트 추정 수행
# 이 함수는 모델의 가중치 및 바이어스에 대한 그래디언트 추정값을 반환하며,
# 계산에 사용된 모멘트(moment) 데이터도 함께 반환함
key, subkey = jax.random.split(key, 2)
weight_grads, bias_grads, clamped_moments, free_moments = estimate_kl_grad(
    subkey,
    training_spec,
    nodes,  # 바이어스 그래디언트를 계산할 노드들  
    edges,  # 가중치 그래디언트를 계산할 엣지들    
    [data],
    [],
    init_state_clamped,
    init_state_free,
)
print(f"그래디언트 계산 완료.")

print(weight_grads)
print(bias_grads)

#mesh = jax.make_mesh((1,), ("x",))                     # <- 8개 GPU 설정 (삭제)
#sharding = jax.sharding.NamedSharding(mesh, P("x"))    # <- Sharding 불필요 (삭제)

timing_program = IsingSamplingProgram(model, free_blocks, [])

timing_chain_len = 100

#batch_sizes = [8, 80, 800, 8000, 64_000, 160_000, 320_000]  # <- 너무 큼
batch_sizes = [8, 16, 32, 64, 128] # 규모 수정 (랩탑용)

times = []
flips = []
dofs = []

schedule = SamplingSchedule(timing_chain_len, 1, 1)

call_f = jax.jit(
    jax.vmap(lambda k: sample_states(k, timing_program, schedule, [x[0] for x in init_state_free], [], [Block(nodes)]))
)

# 5: 벤치마크 루프
print(f"\n5. 벤치마크 루프 시작... (배치 크기: {batch_sizes})")

for batch_size in batch_sizes:
    # 개별 배치 시작
    print(f"  > 배치 {batch_size} 실행 중...")
    key, subkey = jax.random.split(key, 2)
    keys = jax.random.split(key, batch_size)
    
    #keys = jax.device_put(keys, sharding)     # <- Sharding 불필요 (삭제)
    
    _ = jax.block_until_ready(call_f(keys))

    start_time = time.time()
    _ = jax.block_until_ready(call_f(keys))    # 실제 측정
    stop_time = time.time()

    # 개별 배치 완료
    elapsed_time = stop_time - start_time
    print(f"배치 {batch_size} 완료. (소요 시간: {elapsed_time:.4f} 초)")

    times.append(stop_time - start_time)
    # 총 스텝 수 (100) x 총 노드 수 (400) x 병렬 작업 수 (배치 크기)
    flips.append(timing_chain_len * len(nodes) * batch_size)
    dofs.append(batch_size * len(nodes))

# (총 걸린 시간)을 나노초 단위로 변환
flips_per_ns = [x / (y * 1e9) for x, y in zip(flips, times)]

try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
        capture_output=True, text=True, check=True, encoding='utf-8'
    )
    gpu_name = result.stdout.strip() # 공백 제거
    print(f"GPU 이름 감지 성공: {gpu_name}")
except Exception as e:
    # 명령어 실행에 실패하면 (NVIDIA 드라이버가 없거나, macOS 등)
    print(f"GPU 이름 감지 실패. 기본 제목을 사용합니다. (오류: {e})")
    gpu_name = "My GPU"

fig, axs = plt.subplots()
plt.title(f"Performance on 1x{gpu_name}")

axs.plot(dofs, flips_per_ns)
axs.set_xscale("log")
axs.set_xlabel("Parallel Degrees of Freedom")
axs.set_ylabel("Flips/ns")
plt.savefig("fps.png", dpi=300)

# 그래프 출력
print(f"그래프 생성 완료. 'fps.png' 파일 저장됨. 화면에 그래프를 띄웁니다.")
plt.show()
