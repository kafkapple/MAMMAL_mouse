# Session Continuation Summary - ML Keypoint Detection Manual Labeling Preparation

**날짜**: 2025-11-15
**작업 시간**: ~1시간
**상태**: Manual Labeling 준비 완료, 다음 단계로 진행 준비됨

---

## ✅ 이번 세션 완료 작업

### 1. 이전 세션 컨텍스트 복구 (100% 완료)

**작업 내용**:
- 2025-11-14 세션에서 진행된 ML Keypoint Detection 통합 작업 전체 이해
- Phase 1: YOLOv8-Pose Infrastructure 완료 상태 확인
- Phase 2: SuperAnimal Integration 90% 완료, DLC API 이슈 확인
- Phase 3: Manual Labeling 준비 완료 확인

**주요 발견사항**:
- YOLOv8 학습: mAP ~0 (geometric labels 품질 문제로 예상된 실패)
- SuperAnimal: DLC 2.3.11 TensorFlow API는 단일 이미지 미지원, DLC 3.0 PyTorch 필요
- Geometric fallback: 15/22 keypoints, conf=0.5로 예상보다 양호한 성능
- **Manual labeling이 가장 현실적이고 효과적인 접근**으로 결론

### 2. Manual Labeling 워크플로우 문서화 (100% 완료)

**생성된 문서**:
1. `QUICK_START_LABELING.md` (307 lines)
   - 전체 워크플로우: 라벨링 → 학습 → 평가 → 통합
   - 예상 시간: 총 3-4시간 (라벨링 2-3시간, 학습 30분)
   - 예상 개선: mAP 0→0.6-0.8, Paw detection 0%→70-80%

2. `docs/ROBOFLOW_LABELING_GUIDE.md` (263 lines)
   - Roboflow 특화 가이드
   - 22 keypoints 정확한 순서 정의
   - Setup → Labeling → Export → Validation → Training 전체 과정
   - 상세한 keypoint placement 가이드

3. `docs/MANUAL_LABELING_GUIDE.md` (이전 세션 생성)
   - 일반적인 manual labeling 가이드
   - 다양한 도구 옵션 (Roboflow, Label Studio, CVAT)

**핵심 도구**:
- `sample_images_for_labeling.py`: 20개 이미지 샘플링 완료
- `preprocessing_utils/visualize_yolo_labels.py`: 라벨 검증용 시각화

### 3. Manual Labeling 데이터셋 준비 (100% 완료)

**샘플링 결과**:
- 위치: `data/manual_labeling/images/`
- 파일: `sample_000.png` ~ `sample_019.png` (20개)
- 크기: 각 57KB ~ 82KB
- 상태: 라벨링 준비 완료 ✅

**디렉토리 구조**:
```
data/manual_labeling/
├── images/           # 20 images ✅
├── masks/            # 20 masks ✅
├── labels/           # (라벨링 후 생성 예정)
└── viz/              # (검증 시각화 저장 예정)
```

### 4. 22 Keypoint 정의 표준화 (100% 완료)

**MAMMAL 22 Keypoints 순서 (Critical: 순서 정확히 지킬 것!)**:
```
Head (0-5):
  0: nose, 1: left_ear, 2: right_ear, 3: left_eye,
  4: right_eye, 5: head_center

Spine (6-13):
  6-13: spine_1 to spine_8 (neck → tail base, 균등 분포)

Paws (14-17):
  14: left_front_paw, 15: right_front_paw,
  16: left_rear_paw, 17: right_rear_paw

Tail (18-20):
  18: tail_base, 19: tail_mid, 20: tail_tip

Body (21):
  21: centroid
```

---

## 📊 예상 개선 효과

### Before (Geometric Baseline)
```
Detected: 15/22 keypoints
Confidence: 0.40-0.60
Paw detection: 0%
Loss: ~300K
mAP: ~0
```

### After (Manual Labels + Fine-tuned YOLO)
```
Detected: 20-22/22 keypoints
Confidence: 0.80-0.95 (2× improvement)
Paw detection: 70-80%
Loss: 15K-30K (10-20× improvement)
mAP: 0.6-0.8
```

---

## 🎯 다음 즉시 실행 단계 (Recommended Workflow)

### Step 1: Roboflow 계정 및 프로젝트 생성 (5분)

1. https://roboflow.com/ 접속 및 가입
2. "Create New Project" 클릭
3. Project Type: **Keypoint Detection** 선택
4. Project Name: `MAMMAL_Mouse_Keypoints`

### Step 2: 22 Keypoints 정의 (2분)

**중요**: 정확한 순서로 입력!

```
0:  nose
1:  left_ear
2:  right_ear
3:  left_eye
4:  right_eye
5:  head_center
6:  spine_1
7:  spine_2
8:  spine_3
9:  spine_4
10: spine_5
11: spine_6
12: spine_7
13: spine_8
14: left_front_paw
15: right_front_paw
16: left_rear_paw
17: right_rear_paw
18: tail_base
19: tail_mid
20: tail_tip
21: centroid
```

### Step 3: 이미지 업로드 (2분)

```bash
# Roboflow Upload UI에서:
# 1. "Upload" → "Upload Images" 클릭
# 2. 다음 경로에서 모든 20개 이미지 선택:
#    /home/joon/dev/MAMMAL_mouse/data/manual_labeling/images/
# 3. 업로드 완료 대기
```

### Step 4: 라벨링 (2-3시간)

**라벨링 팁**:
- 이미지당 약 5-10분 소요
- 5개 이미지마다 5분 휴식
- 줌 인하여 정밀하게 배치
- Mask를 참고로 활용 가능 (필수 아님)

**품질 체크리스트 (각 이미지마다)**:
- [ ] 22개 keypoints 모두 배치
- [ ] Left/right 혼동 없음
- [ ] Spine 8개가 자연스럽게 균등 분포
- [ ] Paws는 관절 중심에 배치 (발끝 아님)
- [ ] Tail은 자연스러운 곡선 따름

### Step 5: Export 및 검증 (5분)

```bash
# Roboflow에서:
# 1. "Generate" 버전 생성
# 2. Export Format: "YOLO v8" 선택
# 3. ZIP 다운로드

# 터미널에서:
cd ~/Downloads
unzip roboflow.zip -d ~/dev/MAMMAL_mouse/data/manual_labeling/roboflow_export

# 라벨 복사
cp -r data/manual_labeling/roboflow_export/train/labels/* \
      data/manual_labeling/labels/

# 라벨 검증 시각화 (첫 5개)
~/miniconda3/envs/mammal_stable/bin/python \
  preprocessing_utils/visualize_yolo_labels.py \
  --images data/manual_labeling/images \
  --labels data/manual_labeling/labels \
  --output data/manual_labeling/viz \
  --max_images 5

# 결과 확인
ls data/manual_labeling/viz/
```

### Step 6: 데이터셋 병합 (2분)

```bash
# Manual (20) + Geometric (50) = 70 total
python preprocessing_utils/merge_datasets.py \
  --manual data/manual_labeling \
  --geometric data/yolo_mouse_pose \
  --output data/yolo_mouse_pose_enhanced \
  --train_split 0.8

# 결과: 56 train + 14 val
```

### Step 7: YOLOv8 Fine-tuning (30분)

```bash
# Enhanced dataset로 fine-tune
~/miniconda3/envs/mammal_stable/bin/python scripts/train_yolo_pose.py \
  --data data/yolo_mouse_pose_enhanced/data.yaml \
  --epochs 100 \
  --batch 8 \
  --imgsz 256 \
  --weights yolov8n-pose.pt \
  --name mammal_mouse_finetuned

# 학습 모니터링 (다른 터미널)
tail -f /tmp/yolo_train.log
```

### Step 8: 평가 및 비교 (10분)

```bash
# Validation set 평가
~/miniconda3/envs/mammal_stable/bin/python -c "
from ultralytics import YOLO

model = YOLO('runs/pose/mammal_mouse_finetuned/weights/best.pt')
metrics = model.val(data='data/yolo_mouse_pose_enhanced/data.yaml')

print(f'mAP50: {metrics.box.map50:.3f}')
print(f'mAP50-95: {metrics.box.map:.3f}')
"

# Geometric vs YOLO 시각적 비교
~/miniconda3/envs/mammal_stable/bin/python fit_monocular.py \
  --input_dir data/manual_labeling/images \
  --output_dir results/yolo_finetuned \
  --detector yolo \
  --yolo_weights runs/pose/mammal_mouse_finetuned/weights/best.pt \
  --max_images 5
```

### Step 9: Production 통합 (5분)

```bash
# Best model을 models/ 디렉토리로 복사
cp runs/pose/mammal_mouse_finetuned/weights/best.pt \
   models/yolo_mouse_pose_finetuned.pt

# fit_monocular.py 기본값 업데이트 (optional)
# --detector 기본값을 'yolo'로
# --yolo_weights 기본값을 'models/yolo_mouse_pose_finetuned.pt'로
```

---

## 📁 참조 문서 및 파일

### 문서
- **Quick Start**: `QUICK_START_LABELING.md`
- **Roboflow Guide**: `docs/ROBOFLOW_LABELING_GUIDE.md`
- **General Guide**: `docs/MANUAL_LABELING_GUIDE.md`
- **이전 세션**: `docs/reports/251114_session_summary.md`

### 코드
- **샘플링**: `sample_images_for_labeling.py`
- **시각화**: `preprocessing_utils/visualize_yolo_labels.py`
- **학습**: `train_yolo_pose.py`
- **Detector**: `preprocessing_utils/yolo_keypoint_detector.py`
- **통합**: `fit_monocular.py`

### 데이터
- **샘플 이미지**: `data/manual_labeling/images/` (20개 준비 완료 ✅)
- **샘플 마스크**: `data/manual_labeling/masks/` (20개)
- **Geometric 데이터**: `data/yolo_mouse_pose/` (50 train, 10 val)

---

## 🔧 Troubleshooting

### Roboflow 접속 안됨
- 인터넷 연결 확인
- 다른 브라우저 시도
- VPN 비활성화

### Export 형식 오류
- **반드시 YOLO v8 선택** (v5, v7 아님)
- 각 .txt 파일: 1 class + 4 bbox + 66 values (22×3 keypoints)

### 학습 실패
- CUDA 확인: `python -c "import torch; print(torch.cuda.is_available())"`
- Batch size 감소: `--batch 4`
- data.yaml 경로 확인

### mAP 낮음 (<0.3)
- 라벨 품질 재확인 (시각화로 검증)
- Epochs 증가: `--epochs 200`
- 추가 이미지 라벨링 (10개 더)

---

## 💡 핵심 교훈 (이전 세션에서)

### 1. Data Quality > Algorithm
- Geometric keypoints로 YOLOv8 학습 → 완전 실패 (mAP ~0)
- **교훈**: ML 모델은 학습 데이터 품질에 절대 의존
- **해결**: Manual labeling이 유일한 현실적 해결책

### 2. Pretrained Models의 한계
- SuperAnimal-TopViewMouse: 좋은 모델이지만 API 제약
- DLC 2.3.11: 단일 이미지 inference 미지원
- DLC 3.0 PyTorch: 아직 정식 릴리스 안됨
- **교훈**: 좋은 도구도 실용성이 중요

### 3. Manual Labeling의 ROI
- 투자: 2-3시간 (20 images × 5-10 min)
- 예상 수익:
  - Confidence 2배 향상 (0.5 → 0.85+)
  - Loss 10-20배 감소 (300K → 15-30K)
  - Paw detection 0% → 70-80%
- **ROI**: 매우 높음 (시간 대비 성능 개선)

### 4. Progressive Workflow
1. Geometric baseline (완료) → 빠르게 PoC 검증
2. Pretrained models 탐색 (완료) → 한계 발견
3. Manual labeling (다음) → 실용적 해결책
4. Fine-tuning (예정) → 최종 성능 달성

---

## ✅ Success Criteria

라벨링 및 학습 완료 후 확인 사항:

- [ ] 20개 이미지, 각 22 keypoints 라벨링 완료
- [ ] 시각화로 라벨 품질 검증 완료
- [ ] YOLOv8 학습 100 epochs 완료
- [ ] **mAP > 0.6 달성**
- [ ] **Paw detection 작동** (confidence > 0.7)
- [ ] fit_monocular.py에 통합
- [ ] 새 이미지에서 테스트 성공

---

## 🚀 이후 개선 계획 (Optional)

1. **라벨 30개 더 추가** → mAP 0.8+ 목표
2. **Hyperparameter 튜닝** → Epochs, batch size, augmentation
3. **Augmentation 강화** → Rotation, flip, scale, mosaic
4. **ONNX Export** → 빠른 inference
5. **Ensemble** → Geometric + YOLO 조합하여 robustness

---

## 📈 Timeline

| 단계 | 예상 시간 | 상태 |
|------|----------|------|
| Roboflow 설정 | 5분 | 대기 중 |
| 22 keypoints 정의 | 2분 | 대기 중 |
| 이미지 업로드 | 2분 | 대기 중 |
| 라벨링 (20 images) | 2-3시간 | 대기 중 |
| Export & 검증 | 5분 | 대기 중 |
| 데이터셋 병합 | 2분 | 대기 중 |
| YOLOv8 학습 | 30분 | 대기 중 |
| 평가 & 통합 | 15분 | 대기 중 |
| **총 예상 시간** | **~3-4시간** | |

---

## 🎯 결론

**현재 상태**:
- ✅ Manual labeling 완벽 준비 (20 images sampled, guides created, tools ready)
- ✅ 전체 워크플로우 문서화 완료
- ✅ 예상 개선 효과 명확 (mAP 0→0.6-0.8, confidence 2×, loss 10-20×)

**다음 행동**:
1. Roboflow 접속 및 프로젝트 생성
2. 22 keypoints 정의 (정확한 순서!)
3. 20 images 라벨링 시작 (2-3시간)

**예상 결과**:
- 오늘 라벨링 완료 시, 내일 학습 및 평가 완료 가능
- 2일 내 production-ready ML keypoint detector 확보

**Ready to start!** 🎯

---

**작성**: 2025-11-15
**작성자**: Claude Code Session Continuation
**다음 단계**: Roboflow 라벨링 시작 → Fine-tuning → Production 통합
