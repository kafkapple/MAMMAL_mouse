# MAMMAL Paper Reference

> 논문 핵심 설정 및 인용

---

## Citation

```bibtex
@article{an2023mammal,
  title={Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL},
  author={An, Liang and Ren, Jiahui and Yu, Tao and Hai, Tang and Jia, Yichang and Liu, Yebin},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={7727},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

**DOI**: https://doi.org/10.1038/s41467-023-43483-w

---

## Paper Key Settings

### Optimization Iterations

> "5 iterations per frame yielded fairly good results for T > 0"

| 단계 | 논문 권장 | 설명 |
|------|----------|------|
| step0 | 60 | 첫 프레임 초기화 |
| step1 | 3-5 | 트래킹 (T > 0) |
| step2 | 3 | 리파인먼트 |

### Silhouette Loss

> "For faster inference, we set wsil=0 to disable silhouette loss and set optimization iterations to 3 during tracking."

| 설정 | 값 | 효과 |
|------|-----|------|
| wsil | 0.0 | 실루엣 Loss 비활성화 |
| 이유 | 속도 향상 | 트래킹 시 불필요 |

### Speed Benchmark

| 단계 | 시간 | 하드웨어 |
|------|------|----------|
| Detection | 50ms/frame | GPU |
| Matching | 0.15ms/frame | CPU |
| **Mesh Fitting** | **1.2-2초/frame** | GPU |

---

## Implementation: paper_fast

논문 설정을 구현한 Config: `conf/optim/paper_fast.yaml`

```yaml
optim:
  solve_step0_iters: 60   # 첫 프레임 (논문: 60)
  solve_step1_iters: 5    # 트래킹 (논문: 3-5)
  solve_step2_iters: 3    # 리파인먼트 (논문: 3)

loss_weights:
  mask_step1: 0.0   # wsil=0
  mask_step2: 0.0
```

### 예상 시간

| 프레임 수 | 시간 |
|-----------|------|
| 100 | ~15분 |
| 3,600 | ~10시간 |

---

## Usage

```bash
./run_experiment.sh baseline_6view_keypoint frames=aligned_posesplatter optim=paper_fast
```

---

## Related Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) - 최적화 가이드
- [DATASET.md](DATASET.md) - 데이터셋 스펙

---

*Last updated: 2026-02-06*
