# 🎯 SAM Annotation 시작하기

## ✅ 서버가 이미 실행 중입니다!

SAM Annotator가 백그라운드에서 실행 중입니다.

### 접속 방법

#### 로컬에서 접속
브라우저에서 열기:
```
http://localhost:7860
```

#### 원격에서 접속 (SSH 터널)
로컬 PC 터미널에서:
```bash
ssh -L 7860:localhost:7860 joon@bori
```

그 다음 로컬 브라우저에서:
```
http://localhost:7860
```

---

## 서버 관리 명령어

### 서버 상태 확인
```bash
curl -s http://localhost:7860 > /dev/null && echo "✅ Running" || echo "❌ Not running"
```

### 서버 중지
```bash
# 포트 7860 사용 프로세스 종료
lsof -ti :7860 | xargs kill -9

# 또는 SAM 관련 모든 프로세스 종료
pkill -f "run_sam_gui"
```

### 서버 재시작
```bash
# 1. 기존 프로세스 종료
lsof -ti :7860 | xargs kill -9

# 2. 새로 시작
conda activate mammal_stable
cd /home/joon/dev/MAMMAL_mouse

python run_sam_gui.py \
    --frames-dir data/100-KO-male-56-20200615_frames \
    --port 7860
```

### 백그라운드로 실행 (터미널 닫아도 계속)
```bash
conda activate mammal_stable
cd /home/joon/dev/MAMMAL_mouse

nohup python run_sam_gui.py \
    --frames-dir data/100-KO-male-56-20200615_frames \
    --port 7860 \
    > sam_annotator.log 2>&1 &

# 로그 확인
tail -f sam_annotator.log
```

---

## 어노테이션 작업 순서

1. **브라우저에서 http://localhost:7860 접속**

2. **첫 프레임 로드**
   - 슬라이더: 0
   - "📂 Load Frame" 클릭

3. **포인트 추가**
   - "Foreground" 선택
   - 생쥐 위 3-5곳 클릭 (초록 점)
   - "Background" 선택
   - 배경 1-2곳 클릭 (빨간 점)

4. **마스크 생성**
   - "🎯 Generate Mask" 클릭
   - 오른쪽에 마스크 확인

5. **저장**
   - 마스크가 괜찮으면: "💾 Save Annotation"
   - 이상하면: "🗑️ Clear" → 다시 시도

6. **다음 프레임**
   - 슬라이더: 1
   - "📂 Load Frame" 클릭
   - 반복...

---

## 팁

### 빠른 작업
- Foreground 3-4개 + Background 1-2개가 기본
- 마스크 잘 나오면 바로 저장
- 프레임당 30초-1분 목표

### 문제 해결
- **마스크 이상함**: Clear → 포인트 재배치
- **생쥐 일부 빠짐**: Foreground 포인트 추가
- **배경 많이 포함**: Background 포인트 추가

### 진행 상황
- 총 20개 프레임
- 예상 시간: 20-30분
- 중간에 멈춰도 됨 (이어서 가능)

---

## 현재 상태

- ✅ 서버 실행 중: http://localhost:7860
- ✅ 프레임 준비 완료: 20개
- 📂 프레임 위치: `data/100-KO-male-56-20200615_frames/`
- 💾 저장 위치: `data/100-KO-male-56-20200615_frames/annotations/`

---

**지금 바로 http://localhost:7860 접속해서 시작하세요!** 🚀
