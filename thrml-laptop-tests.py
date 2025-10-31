import streamlit as st
import subprocess
import time

# --- 이 아래는 댁의 벤치마크 스크립트에서 가져온 것 ---
import jax
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from thrml.block_management import Block
from thrml.block_sampling import sample_states, SamplingSchedule
from thrml.models.ising import IsingSamplingProgram, IsingEBM
from thrml.pgm import SpinNode

# --- GPU 이름 가져오기 ---
# @st.cache_data : 이 함수는 한 번만 실행하고 결과를 '캐시(저장)'
#                  페이지를 새로고침해도 매번 nvidia-smi를 호출하지 않아 효율적
@st.cache_data
def get_gpu_name():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        gpu_name = result.stdout.strip()
        return gpu_name
    except Exception as e:
        return "N/A (GPU 감지 실패 또는 NVIDIA 드라이버 없음)"

# --- 벤치마크 실행 함수 ---
def run_benchmark(gpu_name):
    graph = nx.grid_graph(dim=(20, 20))
    
    coord_to_node = {coord: SpinNode() for coord in graph.nodes}
    nx.relabel_nodes(graph, coord_to_node, copy=False)

    nodes = list(graph.nodes)
    edges = list(graph.edges)

    key = jax.random.key(4242)
    key, subkey = jax.random.split(key, 2)
    biases = jax.random.normal(subkey, (len(nodes),))
    key, subkey = jax.random.split(key, 2)
    weights = jax.random.normal(subkey, (len(edges),))
    beta = jnp.array(1.0)
    model = IsingEBM(nodes, edges, biases, weights, beta)
    
    coloring = nx.coloring.greedy_color(graph, strategy="DSATUR")
    n_colors = max(coloring.values()) + 1
    free_coloring = [[] for _ in range(n_colors)]
    for node in graph.nodes:
        free_coloring[coloring[node]].append(node)
    free_blocks = [Block(x) for x in free_coloring]
    
    # 벤치마크 실행
    timing_program = IsingSamplingProgram(model, free_blocks, [])
    timing_chain_len = 100
    batch_sizes = [8, 16, 32, 64, 128] # (랩탑용 축소 버전)
    times, flips, dofs = [], [], []
    schedule = SamplingSchedule(timing_chain_len, 1, 1)

    # JAX 컴파일 및 vmap
    call_f = jax.jit(
        jax.vmap(lambda k: sample_states(k, timing_program, schedule, [x[0] for x in [hinton_init_stub(k, model, free_blocks, (1,))] * len(free_blocks)], [], [Block(nodes)]))
    )
    
    # hinton_init_stub 정의 (init_state_free가 없으므로 임시 생성)
    # 실제로는 hinton_init을 import해야 하지만, 빠른 예제를 위해 간단히 만듦.
    def hinton_init_stub(key, model, blocks, shape):
        # 이 부분은 실제 벤치마크의 init_state_free 생성 로직을 가져와야 함.
        # 간단히 무작위로 대체
        return [jax.random.bernoulli(key, 0.5, (shape[0], len(b.nodes))).astype(jnp.bool) for b in blocks]
    
    init_state_free_stub = hinton_init_stub(jax.random.key(0), model, free_blocks, (1,))
    
    call_f = jax.jit(
        jax.vmap(lambda k: sample_states(k, timing_program, schedule, [x[0] for x in init_state_free_stub], [], [Block(nodes)]))
    )

    # 벤치마크 루프
    progress_bar = st.progress(0, text="벤치마크 준비 중...")
    
    for i, batch_size in enumerate(batch_sizes):
        progress_text = f"배치 {batch_size} 실행 중... ( {i+1} / {len(batch_sizes)} )"
        progress_bar.progress((i+1) / len(batch_sizes), text=progress_text)
        
        key, subkey = jax.random.split(key, 2)
        keys = jax.random.split(key, batch_size)
        
        _ = jax.block_until_ready(call_f(keys)) # 워밍업
        start_time = time.time()
        _ = jax.block_until_ready(call_f(keys)) # 실제 측정
        stop_time = time.time()

        times.append(stop_time - start_time)
        flips.append(timing_chain_len * len(nodes) * batch_size)
        dofs.append(batch_size * len(nodes))
        
    progress_bar.empty() # 프로그레스바 제거

    flips_per_ns = [x / (y * 1e9) for x, y in zip(flips, times)]

    # Matplotlib 그래프 생성
    fig, axs = plt.subplots()
    plt.title(f"Performance on 1x{gpu_name}")
    axs.plot(dofs, flips_per_ns)
    axs.set_xscale("log")
    axs.set_xlabel("Parallel Degrees of Freedom")
    axs.set_ylabel("Flips/ns")
    
    return fig # 그래프 객체(fig)를 반환

# --- Streamlit UI ---

st.title("Extropic `thrml` 벤치마크 시뮬레이터")

# 1번 함수를 실행해서 GPU 이름 가져오기
gpu_name = get_gpu_name()
st.success(f"이 앱은 서버의 **{gpu_name}**에서 실행 중입니다.")

# 벤치마크 실행 버튼
if st.button("내 PC에서 TSU 벤치마크 실행하기"):
    
    # 2번 함수를 실행하고 결과(fig)를 받음
    with st.spinner("JAX 컴파일 및 벤치마크 실행 중... (첫 실행은 1~2분 소요될 수 있습니다)"):
        benchmark_figure = run_benchmark(gpu_name)
    
    st.success("벤치마크 완료!")
    
    # Streamlit에 그래프 표시
    st.pyplot(benchmark_figure)
