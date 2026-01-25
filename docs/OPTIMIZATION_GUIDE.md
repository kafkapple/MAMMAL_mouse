# MAMMAL Mesh Fitting 최적화 가이드

## 참고 문헌
An, L., Ren, J., Yu, T., Hai, T., Jia, Y., & Liu, Y. (2023). Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL. *Nature Communications*, 14(1), 7727.
https://doi.org/10.1038/s41467-023-43483-w

---

## 논문 핵심 설정

### Optimization Iterations
| 단계 | 논문 권장 | 기본값 | 비고 |
|------|----------|--------|------|
| step0 (초기화) | 60 | 10 | 첫 프레임만 |
| step1 (tracking) | **3-5** | 100 | T > 0 |
| step2 (refinement) | **3** | 30 | |

### 속도 벤치마크 (논문)
- Detection: 50ms/frame (GPU)
- Matching: 0.15ms/frame (CPU)
- **Mesh Fitting: 1.2-2초/frame** (GPU)

### 빠른 추론 설정 (논문)
> "For faster inference, we set wsil=0 to disable silhouette loss and set optimization iterations to 3 during tracking."

---

## Config 파일 구조

### Optimization Configs (`conf/optim/`)
| Config | step0 | step1 | step2 | render | 용도 |
|--------|-------|-------|-------|--------|------|
| default | 10 | 100 | 30 | ✅ | 기본 (느림) |
| fast | 10 | 50 | 15 | ✅ | 빠른 테스트 |
| paper | 60 | 5 | 3 | ✅ | 논문 설정 + 렌더링 |
| **paper_fast** | **60** | 5 | 3 | ❌ | **논문 설정 + 최고속** ★ |

> **paper_fast**: 논문과 동일한 iteration, 렌더링은 후처리로 대체 → 최고 속도

### Frame Configs ()
| Config | 프레임 수 | 용도 |
|--------|-----------|------|
| quick_test_30 | 30 | 빠른 검증 |
| aligned_test_100 | 100 | 표준 테스트 |
| medium_500 | 500 | 중간 규모 |
| aligned_posesplatter | 3600 | 전체 (pose-splatter 정렬) |

---

## 예상 시간

| Config | 100 프레임 | 3600 프레임 |
|--------|-----------|-------------|
| default | ~20시간 | ~31일 |
| paper | ~1.4시간 | ~2일 |
| **turbo** | **~15분** | **~10시간** |

---

## 권장 실행 명령

### 1. 피팅 실행
```bash
cd /home/joon/dev/MAMMAL_mouse

# ★ 논문 설정 + 전체 (~10시간) - 권장
nohup ./run_experiment.sh baseline_6view_keypoint frames=aligned_posesplatter optim=paper_fast \
  > logs/fitting_paper_fast_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 테스트용
./run_experiment.sh baseline_6view_keypoint frames=aligned_test_100 optim=paper_fast
```

### 2. 피팅 완료 후 시각화
```bash
# 샘플 프레임 렌더링
python -m visualization.mesh_visualizer \
    --result_dir results/fitting/<exp_dir> \
    --start_frame 0 --end_frame 1 \
    --save_video --no_rrd

# 전체 시퀀스 비디오
python -m visualization.mesh_visualizer \
    --result_dir results/fitting/<exp_dir> \
    --view_modes orbit fixed \
    --save_video --save_rrd
```

---

## 기존 피팅 결과 현황

| 날짜 | interval | 완료 | 상태 |
|------|----------|------|------|
| 20251206 | 1 | 100 | ⚠️ interval 불일치 |
| 20251213 | 5 | 3 | ❌ 중단 |
| 20260118 | 5 | 7 | ❌ 중단 |

**주의**: 20251206 결과는 interval=1로 피팅되어 pose-splatter (interval=5)와 프레임 인덱스 불일치.

---

*Created: 2026-01-18*
*pose-splatter 프로젝트 연동용*
