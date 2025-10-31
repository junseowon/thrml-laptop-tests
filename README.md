# Extropic `thrml` 라이브러리 GPU 벤치마크

이 프로젝트는 Extropic의 TSU(열역학적 샘플링 장치) 시뮬레이터인 `thrml` 라이브러리의 성능을 개인용 랩탑 GPU 환경에서 테스트합니다.

---

## 🚀 원본 프로젝트 (Original Source)

* **Extropic `thrml` 깃허브:** [https://github.com/extropic-ai/thrml](https://github.com/extropic-ai/thrml)
* **기반 예제 코드:** `Spin-Models-in-THRML.ipynb`

---

## 🛠️ 주요 수정 사항

원본 코드는 8대의 B200(슈퍼컴퓨터) 환경을 가정하고 있어, 개인 랩탑에서 실행할 수 있도록 다음과 같이 수정했습니다.

1.  **그래프 축소:** `dwave_networkx.pegasus_graph(14)` 대신 `nx.grid_graph(dim=(20, 20))`을 사용하여 그래프 크기를 (수천 개 $\rightarrow$ 400개 노드)로 축소했습니다.
2.  **멀티 GPU 코드 제거:** 단일 GPU 환경에 맞게 `jax.sharding` 관련 코드를 모두 삭제했습니다.
3.  **벤치마크 규모 축소:** `batch_sizes`를 `[8, 16, 32, 64, 128]`로 수정하여 랩탑에서 테스트 가능한 범위로 조정했습니다.
4.  **GPU 이름 자동 감지:** `subprocess`와 `nvidia-smi`를 사용하여 Matplotlib 그래프 제목에 현재 실행 중인 GPU의 모델명이 자동으로 표시되도록 수정했습니다.

---

## 📊 벤치마크 결과

이 코드를 서로 다른 두 환경에서 실행한 결과입니다.

### 1. 랩탑: NVIDIA GeForce RTX 4060 Laptop GPU

* **결과:** "서멀 스로틀링(Thermal Throttling)"으로 인해 성능 그래프가 불안정하게 널뛰는 M자형 곡선이 나타났습니다. (최대 약 0.05 Flips/ns)
<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/ad86063f-cff4-4471-8940-68bef3a638f4" />

### 2. 클라우드: Google Colab (Tesla T4)

* **결과:** 강력한 냉각 시스템 덕분에 서멀 스로틀링 없이, 작업량이 늘어남에 따라 성능이 정직하게 증가하는 깨끗한 J-커브를 보여주었습니다. (최대 약 1.45 Flips/ns)
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/41d439a0-e8c8-45a8-aad1-54a2bd0532a7" />

---

## 💡 결론

`thrml` 라이브러리는 JAX를 통해 GPU의 대규모 병렬 처리를 매우 잘 활용합니다. 하지만 랩탑 환경에서는 GPU 자체의 순수 성능보다 **'서멀 스로틀링(발열 제어)'**이 벤치마크 결과에 훨씬 더 큰 영향을 미친다는 것을 확인했습니다.

---

