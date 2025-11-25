# SAM Annotator 실행 가이드

## 문제: conda run과 Hydra 충돌

`conda run`과 Hydra가 충돌하여 `python -m sam_annotator` 방식이 작동하지 않습니다.

**해결책**: Python 스크립트를 직접 실행합니다.

## 실행 방법

### 방법 1: 직접 Python 스크립트 실행 (권장)

```bash
# 1. Conda 환경 활성화
conda activate mammal_stable

# 2. SAM GUI 실행
python run_sam_gui.py \
    --frames-dir data/100-KO-male-56-20200615_frames \
    --port 7860
```

### 방법 2: 백그라운드 실행

```bash
conda activate mammal_stable

# 백그라운드로 실행
nohup python run_sam_gui.py \
    --frames-dir data/100-KO-male-56-20200615_frames \
    --port 7860 \
    > sam_annotator.log 2>&1 &

# 로그 확인
tail -f sam_annotator.log
```

## 웹 UI 접속

### 로컬 접속
```
http://localhost:7860
```

### 원격 접속 (SSH 터널)
```bash
# 로컬 PC에서:
ssh -L 7860:localhost:7860 joon@bori

# 브라우저에서:
http://localhost:7860
```

## 어노테이션 워크플로우

1. **Load Frame**
   - 슬라이더로 프레임 선택
   - "📂 Load Frame" 클릭

2. **Add Points**
   - "Foreground" 선택 → 생쥐 위 클릭 (초록 점)
   - "Background" 선택 → 배경 클릭 (빨간 점)
   - 최소 3-5개 포인트 추가 권장

3. **Generate Mask**
   - "🎯 Generate Mask" 클릭
   - 마스크 확인

4. **Save**
   - 만족스러우면 "💾 Save Annotation" 클릭
   - 다음 프레임으로 이동

5. **Repeat**
   - 모든 프레임에 대해 반복

## 저장 결과

어노테이션 결과는 다음 위치에 저장됩니다:

```
data/100-KO-male-56-20200615_frames/annotations/
├── frame_000000_annotation.json
├── frame_000000_mask.png
├── frame_000001_annotation.json
├── frame_000001_mask.png
└── ...
```

## 다음 단계

어노테이션 완료 후:

```bash
# 크롭된 프레임 생성
conda activate mammal_stable

python process_annotated_frames.py \
    data/100-KO-male-56-20200615_frames/annotations \
    --output-dir data/100-KO-male-56-20200615_cropped \
    --padding 50
```

## 문제 해결

### 포트가 이미 사용 중

```bash
# 프로세스 확인
lsof -i :7860

# 종료
kill -9 <PID>

# 또는 다른 포트 사용
python run_sam_gui.py --frames-dir ... --port 8080
```

### SAM 체크포인트 없음

```bash
cd ~/dev/segment-anything-2/checkpoints
./download_ckpts.sh
```

### GPU 메모리 부족

```bash
# run_sam_gui.py 수정하여 작은 모델 사용
# 또는 CPU 모드로 실행 (느림)
```

## 팁

- **좋은 어노테이션을 위해**:
  - Foreground: 생쥐 머리, 몸통, 꼬리에 3-5개 점
  - Background: 바닥이나 벽에 1-2개 점
  - 경계선에서 멀리 떨어진 명확한 영역에 클릭

- **속도 향상**:
  - 모든 프레임이 아닌 일부만 어노테이션 (10-15개)
  - 2-3 프레임마다 skip 가능

- **저장 습관**:
  - 각 프레임 어노테이션 후 즉시 저장
  - 브라우저 닫기 전 모든 작업 저장 확인

---

**현재 상태**: 20개 프레임 추출 완료
**다음 작업**: SAM 어노테이션 진행
**최종 목표**: Mesh fitting을 위한 크롭된 프레임 생성
