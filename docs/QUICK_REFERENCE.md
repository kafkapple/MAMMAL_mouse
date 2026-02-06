# Quick Reference

> 자주 사용하는 명령어 모음

---

## Mesh Fitting

### Multi-View (기본)

```bash
# 디버그 (5 frames)
./run_mesh_fitting_default.sh quick_test

# 테스트 (100 frames, 논문 설정)
./run_experiment.sh baseline_6view_keypoint frames=aligned_test_100 optim=paper_fast

# 전체 (3,600 frames, ~10시간)
./run_experiment.sh baseline_6view_keypoint frames=aligned_posesplatter optim=paper_fast

# Silhouette only (keypoint 없이)
./run_mesh_fitting_default.sh - 0 10 -- --keypoints none

# 커스텀 데이터
./run_mesh_fitting_default.sh - 0 10 -- --input_dir /path/to/data
```

### Monocular (단일 카메라)

```bash
# 기본
./run_mesh_fitting_monocular.sh data/frames/ results/monocular/output

# 프레임 수 제한
./run_mesh_fitting_monocular.sh data/frames/ results/monocular/output 10

# Silhouette only
./run_mesh_fitting_monocular.sh data/frames/ results/monocular/output - -- --keypoints none
```

### Python 직접 실행

```bash
export PYOPENGL_PLATFORM=egl  # headless 서버 필수

# Multi-view
python fitter_articulation.py \
    dataset=default_markerless optim=fast fitter.end_frame=10

# Monocular
python fit_monocular.py \
    --input_dir data/frames/ --output_dir results/monocular/test \
    --detector geometric --max_images 10
```

---

## Preprocessing

### 프레임 추출

```bash
# 비디오에서 프레임 추출 (10fps)
ffmpeg -i video.mp4 -vf fps=10 frames/frame_%04d.png

# 전체 프레임 (원본 fps)
ffmpeg -i video.mp4 frames/%06d.png
```

### SAM 마스크 생성

```bash
python preprocessing_utils/sam_inference.py \
    --input_dir frames/ --output_dir masks/
```

### 전처리 파이프라인

```bash
python scripts/preprocess.py \
    dataset=my_video mode=single_view_preprocess
```

---

## Visualization

### mesh_visualizer

```bash
# 샘플 렌더링
python -m visualization.mesh_visualizer \
    --result_dir results/fitting/<exp> \
    --start_frame 0 --end_frame 1 \
    --save_video --no_rrd

# 전체 시퀀스
python -m visualization.mesh_visualizer \
    --result_dir results/fitting/<exp> \
    --view_modes orbit fixed \
    --save_video --save_rrd

# OBJ 직접 (BodyModel 불필요)
python scripts/visualize_mesh_sequence.py results/fitting/<exp> --use-obj -o mesh.mp4

# 360도 회전
python scripts/visualize_mesh_sequence.py results/fitting/<exp> --rotating -o rotating.mp4
```

### 실험 비교

```bash
python scripts/compare_experiments.py \
    results/fitting/exp1_* results/fitting/exp2_* \
    --output comparison.html
```

---

## Config 주요 설정

### Optimization Iterations

| 설정 | fast | default | paper_fast | accurate |
|------|------|---------|------------|----------|
| step0_iters | 10 | 10 | 60 | 20 |
| step1_iters | 50 | 100 | 5 | 200 |
| step2_iters | 15 | 30 | 3 | 50 |

### Loss Weights

| Loss | Weight | 설명 |
|------|--------|------|
| theta | 3.0 | 관절 정규화 |
| 2d | 0.2 | Keypoint reprojection |
| bone | 0.5 | 뼈 길이 제약 |
| mask | 0/3000 | 실루엣 (Step2에서만) |
| stretch | 1.0 | Stretch 페널티 |
| temp | 0.25 | 시간적 부드러움 |

### Silhouette-Only 설정

```bash
# 권장 설정
--keypoints none \
    silhouette.iter_multiplier=3.0 \
    silhouette.theta_weight=15.0
```

---

## 환경

```bash
# 환경 활성화
conda activate mammal_stable

# GPU 확인
nvidia-smi

# PyTorch3D 확인
python -c "import pytorch3d; print(pytorch3d.__version__)"

# EGL 확인 (headless 서버)
ldconfig -p | grep EGL
```

---

## 결과 확인

```bash
# 결과 디렉토리
ls results/fitting/

# OBJ 메쉬 확인
ls results/fitting/<exp>/obj/

# Loss 기록
cat results/fitting/<exp>/loss_history.json | python -m json.tool | tail -20

# 렌더링 이미지
ls results/fitting/<exp>/render/
```

---

## 자동 재시작 (중단된 실험)

```bash
./run_experiment.sh <experiment> --resume_from results/fitting/<result_folder>
```

마지막 완료 프레임 자동 감지, 다음 프레임부터 재시작한다.

---

*Last updated: 2026-02-06*
