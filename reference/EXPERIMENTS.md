# Experiments Guide

> 실험 실행 명령어 및 워크플로우

---

## Quick Reference

```bash
# 전체 피팅 (논문 설정, ~10시간)
./run_experiment.sh baseline_6view_keypoint frames=aligned_posesplatter optim=paper_fast

# 테스트 (100프레임, ~15분)
./run_experiment.sh baseline_6view_keypoint frames=aligned_test_100 optim=paper_fast

# 디버그 (5프레임)
./run_experiment.sh baseline_6view_keypoint --debug
```

---

## Experiment Configs

### Baseline

| Config | Views | Keypoints | 용도 |
|--------|-------|-----------|------|
| baseline_6view_keypoint | 6 | 22 | 논문 기준 |

### Keypoint Ablation (6-view 고정)

| Config | Keypoints | 설명 |
|--------|-----------|------|
| baseline_6view_keypoint | 22 | Full |
| sparse_9kp_dlc | 9 | DeepLabCut style |
| sparse_7kp_mars | 7 | MARS style |
| sparse_5kp_minimal | 5 | nose, ears, neck, tail |
| sixview_sparse_keypoint | 3 | nose, body, tail |
| sixview_no_keypoint | 0 | Silhouette only |

### View Ablation (sparse 3kp 고정)

| Config | Views | IDs |
|--------|-------|-----|
| sixview_sparse_keypoint | 6 | 0,1,2,3,4,5 |
| sparse_5view | 5 | 0,1,2,3,4 |
| sparse_4view | 4 | 0,1,2,3 |
| sparse_3view | 3 | 0,2,4 (diagonal) |
| sparse_2view | 2 | 0,3 (opposite) |

---

## Hydra Overrides

### Frame Configs (`frames=`)

| Config | 프레임 수 | 용도 |
|--------|-----------|------|
| aligned_posesplatter | 3,600 | 전체 (pose-splatter 정렬) |
| aligned_test_100 | 100 | 표준 테스트 |
| quick_test_30 | 30 | 빠른 검증 |

### Optim Configs (`optim=`)

| Config | step0 | step1 | step2 | render |
|--------|-------|-------|-------|--------|
| paper_fast | 60 | 5 | 3 | off |
| paper | 60 | 5 | 3 | on |
| fast | 10 | 50 | 15 | on |
| default | 10 | 100 | 30 | on |

---

## Batch Runs

### Keypoint Ablation

```bash
for exp in baseline_6view_keypoint sparse_9kp_dlc sparse_7kp_mars \
           sparse_5kp_minimal sixview_sparse_keypoint sixview_no_keypoint; do
    ./run_experiment.sh $exp frames=aligned_test_100 optim=paper_fast
done
```

### View Ablation

```bash
for exp in sixview_sparse_keypoint sparse_5view sparse_4view \
           sparse_3view sparse_2view; do
    ./run_experiment.sh $exp frames=aligned_test_100 optim=paper_fast
done
```

---

## Background Execution

```bash
# 백그라운드 실행
nohup ./run_experiment.sh baseline_6view_keypoint frames=aligned_posesplatter optim=paper_fast \
  > logs/fitting_paper_fast_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 진행 확인
tail -f logs/fitting_paper_fast_*.log

# 최신 로그만
tail -f $(ls -t logs/fitting_paper_fast_*.log | head -1)
```

---

## Post-Fitting

### 시각화

```bash
# 샘플 프레임 렌더링
python -m mammal_ext.visualization.mesh_visualizer \
    --result_dir results/fitting/<exp_dir> \
    --start_frame 0 --end_frame 1 \
    --save_video --no_rrd

# 전체 시퀀스 비디오
python -m mammal_ext.visualization.mesh_visualizer \
    --result_dir results/fitting/<exp_dir> \
    --view_modes orbit fixed \
    --save_video --save_rrd
```

### 결과 비교

```bash
python scripts/compare_experiments.py \
    results/fitting/exp1_* results/fitting/exp2_* \
    --output comparison.html
```

---

## Debug vs Full 비교

| Mode | Frames | Step0 | Step1 | Step2 | 예상 시간 |
|------|--------|-------|-------|-------|----------|
| Debug | 2 | 5 | 20 | 10 | ~1분 |
| Full | 100 | 10-20 | 100-180 | 30-50 | ~30분 |

---

## Related Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) - 최적화 설정 상세
- [PAPER.md](PAPER.md) - 논문 설정

---

*Last updated: 2026-02-06*
