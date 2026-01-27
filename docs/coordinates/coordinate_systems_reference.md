---
created: 2026-01-27T15:05:40
modified: 2026-01-27T15:14:16
---
# coordinate_systems_reference

## 1. 주요 3D 좌표계

### 1.1 좌표계 정의

| 좌표계 | Up 축 | Forward 축 | Right 축 | 손잡이 | 대표 소프트웨어 |
|--------|--------|-----------|----------|--------|----------------|
| **OpenGL / Blender** | +Y | -Z | +X | Right | Blender, Three.js |
| **OpenCV** | -Y (down) | +Z | +X | Right | OpenCV, COLMAP |
| **Blender World** | +Z | +Y | +X | Right | Blender (world) |
| **MAMMAL** | **-Y** | +X | +Z | Right | MAMMAL fitting |

> **주의**: Blender는 카메라 좌표(OpenGL, Y-up)와 월드 좌표(Z-up)가 다릅니다.

### 1.2 이론적 조합 수

$$
\text{조합 수} = 6 \text{ (up 축)} \times 4 \text{ (forward 축)} \times 2 \text{ (handedness)} = 48
$$

실제로는 소프트웨어마다 관례가 정해져 있어 몇 가지만 사용됩니다.

---

## 2. MAMMAL 좌표계 분석

### 2.1 실험적 발견

MAMMAL fitted mesh의 정점 분석:

```
Original vertices:
  center = (99.2, 24.1, 35.4) mm
  size   = (115.3, 52.8, 40.6) mm
  max_dim = 115.3 mm (X축 = body length)
```

| 축 | 의미 | 범위 (mm) |
|----|------|-----------|
| X | Body length (head→tail) | ~115 |
| Y | **Height (belly→back)** | ~53 |
| Z | Width (left→right) | ~41 |

### 2.2 Up 방향 판별

MAMMAL의 up 방향이 **-Y**인 근거:

1. Y축이 height 방향 (52.8mm)
2. 정점의 Y 값이 양수 → 등(back)이 +Y → 배(belly)가 -Y
3. PoseSplatter sphere_renderer에서 `global_up = -[0,0,1]` 사용 (auto_orient 후)
4. **실험 검증**: `(x, z, -y)` 변환으로 Blender에서 정립 확인

### 2.3 좌표계 비교

```
MAMMAL (-Y up):        Blender World (Z up):
    +Y (back)              +Z (up)
    |                      |
    |                      |
    +---→ +X (head)        +---→ +X
   /                      /
  +Z (right)             +Y (forward)
```

---

## 3. 좌표 변환

### 3.1 MAMMAL → Blender World

**-Y up → Z up** 변환 (X축 기준 +90° 회전):

$$
\begin{pmatrix} x' \\ y' \\ z' \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & -1 & 0 \end{pmatrix} \begin{pmatrix} x \\ y \\ z \end{pmatrix}
$$

즉:

$$
(x, y, z)_{\text{MAMMAL}} \rightarrow (x, z, -y)_{\text{Blender}}
$$

**코드**:
```python
for v in obj.data.vertices:
    old_y = v.co.y
    old_z = v.co.z
    v.co.y = old_z
    v.co.z = -old_y
```

### 3.2 잘못된 변환들 (실험 기록)

| 시도 | 변환 | 가정 | 결과 |
|------|------|------|------|
| 1 | `(x, -y, -z)` | Y-up, 180° flip | 옆으로 누움 |
| 2 | `(x, -z, y)` | Y-up → Z-up | 뒤집힘 (upside down) |
| **3** | **`(x, z, -y)`** | **-Y up → Z-up** | **정립 (correct)** |

### 3.3 일반 변환 공식

| From → To | 변환 | 회전 행렬 |
|-----------|------|-----------|
| OpenGL (Y-up) → Blender (Z-up) | `(x, -z, y)` | Rx(-90°) |
| OpenCV (-Y down, Z fwd) → Blender (Z-up) | `(x, -z, -y)` | Rx(+90°) + flip |
| **MAMMAL (-Y up) → Blender (Z-up)** | **`(x, z, -y)`** | **Rx(+90°)** |
| Blender (Z-up) → MAMMAL (-Y up) | `(x, -z, y)` | Rx(-90°) |

---

## 4. FaceLift / PoseSplatter 카메라 설정

### 4.1 카메라 파라미터

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| fx, fy | 548.99 ≈ 549 | Focal length (pixels) |
| cx, cy | 256 | Principal point (image center) |
| Image size | 512 × 512 | |
| Distance | 2.7 | Camera-to-origin |
| Elevation | 20° | From XY plane |
| Sensor width | 36mm | Blender default |

### 4.2 Blender Focal Length 변환

$$
f_{\text{mm}} = \frac{f_{\text{px}} \times \text{sensor\_width}}{\text{image\_size}} = \frac{549 \times 36}{512} \approx 38.6 \text{mm}
$$

### 4.3 Orbit Camera 배치 (Blender World, Z-up)

Azimuth $\theta$, Elevation $\phi$, Distance $d$:

$$
\begin{aligned}
x &= d \cos\phi \sin\theta \\
y &= -d \cos\phi \cos\theta \\
z &= d \sin\phi
\end{aligned}
$$

- $\theta = 0$일 때 카메라는 -Y 방향 (Blender 정면)
- Elevation $\phi = 20°$로 약간 위에서 내려다봄

### 4.4 FaceLift 학습 설정

| 항목 | 값 |
|------|-----|
| Total rendered views | 32 |
| Sampled per iteration | 8 (random) |
| Input views | 4 |
| Target views | 4 |
| Azimuth spacing | 11.25° (= 360°/32) |

---

## 5. PoseSplatter auto_orient 좌표계

PoseSplatter `utils.py`의 auto_orient 변환 후 좌표계:

```python
up = -np.load(up_fn)["up"]           # up 방향 (부호 반전)
R_2 = rotation_matrix_between([0,0,1], up)  # Z → up 회전
# 변환 후: viewmats의 world up = -Z 방향
```

| 단계 | World Up | 비고 |
|------|----------|------|
| MAMMAL 원본 | -Y | Fitted mesh 좌표계 |
| auto_orient 후 | -Z (데이터 의존) | sphere_renderer 기준 |
| Turntable (수정 전) | +Y | OpenGL 관례 → 불일치 |
| **Turntable (수정 후)** | **-Z** | 데이터와 일치 |

---

## 6. Quick Reference

### 좌표계 판별 체크리스트

1. **Height 축 확인**: 어느 축이 위-아래?
2. **부호 확인**: +가 위인지 아래인지?
3. **Body axis 확인**: 어느 축이 앞-뒤?
4. **실험 검증**: 알려진 좌표계(Blender Z-up)로 변환 후 시각 확인

### 변환 디버깅

올바른 변환인지 확인하는 방법:
1. Blender에서 mesh 로드
2. 축 표시기(axis helper) 추가
3. 카메라 뷰(Numpad 0)에서 방향 확인
4. 등(back)이 위, 배(belly)가 아래면 정상

---

## 7. Blender 카메라 뷰 테스트 매뉴얼

### 7.1 UI 구조 (기본 레이아웃)

```
┌─────────────────────────────────────┬──────────────┐
│                                     │  Outliner     │ ← 우측 상단
│          3D Viewport                │  (씬 계층)    │
│                                     ├──────────────┤
│                                     │  Properties   │ ← 우측 하단
│                                     │  (속성 패널)  │
└─────────────────────────────────────┴──────────────┘
```

**Outliner**: 씬의 모든 오브젝트(Cam_0~5, Mouse, Sun, X/Y/Z_Axis)가 트리 형태로 표시되는 패널.

### 7.2 카메라 뷰 전환

| 단계 | 동작 | 단축키 |
|------|------|--------|
| 1 | Outliner에서 카메라 클릭 (예: `Cam_2`) | 마우스 클릭 |
| 2 | 선택한 카메라를 활성 카메라로 설정 | **Ctrl + Numpad 0** |
| 3 | 카메라 시점으로 전환 | **Numpad 0** |
| 4 | 3D 뷰로 복귀 | **Numpad 0** (토글) |

**대안**: Properties 패널 → Scene 탭 (카메라 아이콘) → Camera 드롭다운에서 직접 선택

### 7.3 빠른 렌더링

| 방법 | 단축키 | 설명 |
|------|--------|------|
| 단일 렌더 | **F12** | 현재 활성 카메라로 렌더링 |
| Viewport 미리보기 | **Z → Rendered** | 실시간 렌더 프리뷰 |
| 전체 카메라 일괄 렌더 | Python 콘솔 | 아래 스크립트 참조 |

### 7.4 전체 카메라 일괄 렌더링 스크립트

Blender Scripting 탭 → Python Console에 붙여넣기:

```python
import bpy
for i in range(6):  # 또는 range(32)
    bpy.context.scene.camera = bpy.data.objects[f"Cam_{i}"]
    bpy.context.scene.render.filepath = f"/tmp/cam_{i}.png"
    bpy.ops.render.render(write_still=True)
    print(f"Rendered Cam_{i}")
```

### 7.5 스크립트 실행 방법

1. Blender 상단 탭에서 **Scripting** 선택
2. 좌측 Text Editor에서 **New** 클릭
3. `local_test_6cam.py` 코드 붙여넣기
4. **Run Script** (▶ 버튼 또는 Alt+P)
5. 이후 위 일괄 렌더 스크립트를 Python Console에서 실행

### 7.6 Numpad 없는 경우 (노트북)

Blender → Edit → Preferences → Input → **Emulate Numpad** 체크
→ 일반 숫자키 0~9가 Numpad처럼 동작

---

## 8. MAMMAL 32-View 렌더링 실험

### 8.1 실험 설정

| 실험 | Offset | PP 보정 | 목적 |
|------|--------|---------|------|
| **MAMMAL_CENTER** | (0, 0, 0) | No | Baseline (생쥐 중앙) |
| **MAMMAL_OFFSET** | (0.3, 0.2, 0) | No | PP 불일치 재현 |
| **MAMMAL_OFFSET_PP** | (0.3, 0.2, 0) | Yes | PP 보정 효과 검증 |

### 8.2 렌더링 파라미터

| 파라미터 | 값 | 근거 |
|----------|-----|------|
| NUM_VIEWS | 32 | FaceLift 원본 설정 |
| IMAGE_SIZE | 512 x 512 | FaceLift 원본 |
| FX, FY | 548.99 | FaceLift 원본 |
| CX, CY | 256.0 | Image center |
| CAMERA_DISTANCE | 2.7 | FaceLift 정규화 기준 |
| ELEVATION | 20 deg | FaceLift 원본 |
| TARGET_OBJECT_SIZE | 1.5 | 약 5% foreground coverage |
| Azimuth spacing | 11.25 deg | 360 / 32 |
| Coordinate transform | (x, z, -y) | MAMMAL -Y up → Blender Z up |

### 8.3 실행 명령어

```bash
cd /home/joon/dev/FaceLift

blender --background --python mouse_extensions/scripts/blender/render_mammal_32view_v2.py -- \
    --experiment MAMMAL_CENTER \
    --output_dir /home/joon/data/synthetic/MAMMAL_CENTER \
    --num_samples 100
```

### 8.4 출력 구조

```
MAMMAL_CENTER/
├── sample_00000/
│   ├── images/
│   │   ├── cam_000.png  (512x512 RGBA)
│   │   └── ... (32 views)
│   └── opencv_cameras.json
└── data_train.txt
```

### 8.5 FaceLift 학습 데이터 사용

| 단계 | 수량 | 설명 |
|------|------|------|
| 전체 렌더링 | 32 views | Orbit 카메라 |
| 학습 시 샘플링 | 8 views (random) | Per iteration |
| Input views | 4 | Encoder 입력 |
| Target views | 4 | Reconstruction 대상 |

---

## 9. GS-LRM 추론 (Inference)

### 9.1 스크립트 위치

`/home/joon/dev/FaceLift/inference_mouse.py`

### 9.2 입력 포맷

```
sample_dir/
├── images/
│   ├── cam_000.png  (6+ views, 512x512)
│   └── ...
└── opencv_cameras.json
```

`opencv_cameras.json` 구조:
```json
{"frames": [{"fx": 549, "fy": 549, "cx": 256, "cy": 256,
             "w": 512, "h": 512,
             "w2c": [[...]], "c2w": [[...]],
             "file_path": "images/cam_000.png"}, ...]}
```

### 9.3 실행 명령어

```bash
cd /home/joon/dev/FaceLift

# 단일 샘플
python inference_mouse.py \
    --sample_dir /path/to/sample_00000 \
    --checkpoint checkpoints/gslrm/D8_E0_paper_original/ \
    --output_dir outputs/mammal_inference/ \
    --save_turntable --save_mesh

# 다수 샘플 일괄
python inference_mouse.py \
    --data_dir /path/to/MAMMAL_CENTER/ \
    --checkpoint checkpoints/gslrm/D8_E0_paper_original/ \
    --output_dir outputs/mammal_inference/
```

### 9.4 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--sample_dir` | - | 단일 샘플 경로 |
| `--data_dir` | - | 다수 샘플 디렉토리 |
| `--checkpoint` | `checkpoints/gslrm/mouse/` | 체크포인트 디렉토리 |
| `--save_turntable` | True | Turntable 비디오 생성 |
| `--save_mesh` | True | PLY/OBJ 메시 저장 |
| `--use_pretrained_facelift` | False | 원본 FaceLift 가중치 |

---

*Created: 2026-01-28 | MAMMAL -Y up 좌표계 실험 검증 완료*
