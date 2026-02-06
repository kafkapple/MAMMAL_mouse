# Coordinate Systems Reference

> MAMMAL, OpenGL, OpenCV, Blender 좌표계 및 변환, PoseSplatter 정렬

---

## 주요 3D 좌표계

| 좌표계 | Up 축 | Forward 축 | Right 축 | 손잡이 | 대표 소프트웨어 |
|--------|--------|-----------|----------|--------|----------------|
| **OpenGL / Blender Camera** | +Y | -Z | +X | Right | Blender, Three.js |
| **OpenCV** | -Y (down) | +Z | +X | Right | OpenCV, COLMAP |
| **Blender World** | +Z | +Y | +X | Right | Blender (world) |
| **MAMMAL** | **-Y** | +X | +Z | Right | MAMMAL fitting |

> **주의**: Blender는 카메라 좌표(OpenGL, Y-up)와 월드 좌표(Z-up)가 다르다.

---

## MAMMAL 좌표계 분석

### 실험적 발견

MAMMAL fitted mesh의 정점 분석 결과:

```
Original vertices:
  center = (99.2, 24.1, 35.4) mm
  size   = (115.3, 52.8, 40.6) mm
  max_dim = 115.3 mm (X축 = body length)
```

| 축 | 의미 | 범위 (mm) |
|----|------|-----------|
| X | Body length (head -> tail) | ~115 |
| Y | Height (belly -> back) | ~53 |
| Z | Width (left -> right) | ~41 |

### Up 방향이 -Y인 근거

1. Y축이 height 방향 (52.8mm)
2. 정점의 Y 값이 양수 = 등(back)이 +Y = 배(belly)가 -Y
3. PoseSplatter `sphere_renderer`에서 `global_up = -[0,0,1]` (auto_orient 후)
4. 실험 검증: `(x, z, -y)` 변환으로 Blender에서 정립 확인

### 좌표계 비교

```
MAMMAL (-Y up):        Blender World (Z up):
    +Y (back)              +Z (up)
    |                      |
    |                      |
    +----> +X (head)       +----> +X
   /                      /
  +Z (right)             +Y (forward)
```

---

## 좌표 변환

### MAMMAL -> Blender World

**-Y up -> Z up** 변환 (X축 기준 +90도 회전):

```
(x, y, z)_MAMMAL  ->  (x, z, -y)_Blender
```

회전 행렬:

```
| 1  0  0 |
| 0  0  1 |
| 0 -1  0 |
```

**Python 코드**:
```python
# Blender에서
for v in obj.data.vertices:
    old_y = v.co.y
    old_z = v.co.z
    v.co.y = old_z
    v.co.z = -old_y
```

### 잘못된 변환들 (실험 기록)

| 시도 | 변환 | 가정 | 결과 |
|------|------|------|------|
| 1 | `(x, -y, -z)` | Y-up, 180 flip | 옆으로 누움 |
| 2 | `(x, -z, y)` | Y-up -> Z-up | 뒤집힘 |
| **3** | **`(x, z, -y)`** | **-Y up -> Z-up** | **정립 (correct)** |

### 일반 변환 공식

| From -> To | 변환 | 회전 행렬 |
|-----------|------|-----------|
| OpenGL (Y-up) -> Blender (Z-up) | `(x, -z, y)` | Rx(-90) |
| OpenCV (-Y down, Z fwd) -> Blender (Z-up) | `(x, -z, -y)` | Rx(+90) + flip |
| **MAMMAL (-Y up) -> Blender (Z-up)** | **`(x, z, -y)`** | **Rx(+90)** |
| Blender (Z-up) -> MAMMAL (-Y up) | `(x, -z, y)` | Rx(-90) |

---

## PoseSplatter 좌표계 정렬

### auto_orient 변환

PoseSplatter `utils.py`의 auto_orient 로직:

```python
up = -np.load(up_fn)["up"]                    # up 방향 (부호 반전)
R_2 = rotation_matrix_between([0,0,1], up)     # Z -> up 회전
mean_translation = mean(R.T @ translation)     # 카메라 위치 평균
rotation = R @ R_2.T                           # R_2.T 적용
translation = (R @ mean_translation) + translation
scale_factor = 1 / max(norm(positions))        # 정규화
translation *= scale_factor
```

**3D 점 변환 공식**: `P' = scale_factor * (R_2.T @ P)`

### World Up 단계별 변화

| 단계 | World Up | 비고 |
|------|----------|------|
| MAMMAL 원본 | -Y | Fitted mesh 좌표계 |
| auto_orient 후 | -Z (데이터 의존) | sphere_renderer 기준 |
| Turntable (수정 전) | +Y | OpenGL 관례 -> 불일치 |
| **Turntable (수정 후)** | **-Z** | 데이터와 일치 |

### init_mode별 좌표계 매핑

```
init_mode -> coordinate_system -> world_up

shape_carving    -> "original"  -> [0, 1, 0]   (Y-up)
mesh_per_frame   -> "mammal"    -> [0, 0, -1]  (-Z up)
mesh_lbs         -> "mammal"    -> [0, 0, -1]  (-Z up)
keypoint         -> "original"  -> [0, 1, 0]   (Y-up)
```

**핵심 문제**: init_mode에 따라 Gaussian params가 다른 좌표계에 존재하지만 turntable이 단일 좌표계만 사용하면 생쥐가 옆으로 누워 보인다.

**해결**: `src/modules/core/coordinates.py`에서 `get_world_up(coordinate_system)` 자동 감지.

---

## FaceLift / PoseSplatter 카메라 설정

### 카메라 파라미터

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| fx, fy | 548.99 (약 549) | Focal length (pixels) |
| cx, cy | 256 | Principal point (image center) |
| Image size | 512 x 512 | |
| Distance | 2.7 | Camera-to-origin |
| Elevation | 20도 | From XY plane |
| Sensor width | 36mm | Blender default |

### Blender Focal Length 변환

```
f_mm = (f_px * sensor_width) / image_size = (549 * 36) / 512 = 38.6 mm
```

### Orbit Camera 배치 (Blender World, Z-up)

Azimuth theta, Elevation phi, Distance d:

```
x = d * cos(phi) * sin(theta)
y = -d * cos(phi) * cos(theta)
z = d * sin(phi)
```

- theta=0 -> 카메라는 -Y 방향 (Blender 정면)
- phi=20도 -> 약간 위에서 내려다봄

### FaceLift 학습 설정

| 항목 | 값 |
|------|-----|
| Total rendered views | 32 |
| Sampled per iteration | 8 (random) |
| Input views | 4 |
| Target views | 4 |
| Azimuth spacing | 11.25도 (= 360/32) |

---

## MAMMAL 32-View Blender 렌더링

### 렌더링 파라미터

| 파라미터 | 값 |
|----------|-----|
| NUM_VIEWS | 32 |
| IMAGE_SIZE | 512 x 512 |
| CAMERA_DISTANCE | 2.7 |
| ELEVATION | 20도 |
| TARGET_OBJECT_SIZE | 1.5 |
| Azimuth spacing | 11.25도 |
| Coordinate transform | (x, z, -y) (MAMMAL -> Blender) |

### 실행

```bash
blender --background --python render_mammal_32view_v2.py -- \
    --experiment MAMMAL_CENTER \
    --output_dir /home/joon/data/synthetic/MAMMAL_CENTER \
    --num_samples 100
```

---

## Blender 카메라 뷰 테스트

### 카메라 뷰 전환

| 단계 | 동작 | 단축키 |
|------|------|--------|
| 1 | Outliner에서 카메라 클릭 | 마우스 클릭 |
| 2 | 활성 카메라로 설정 | Ctrl + Numpad 0 |
| 3 | 카메라 시점 전환 | Numpad 0 |

### 전체 카메라 일괄 렌더링

```python
import bpy
for i in range(6):
    bpy.context.scene.camera = bpy.data.objects[f"Cam_{i}"]
    bpy.context.scene.render.filepath = f"/tmp/cam_{i}.png"
    bpy.ops.render.render(write_still=True)
```

---

## Quick Reference

### 좌표계 판별 체크리스트

1. **Height 축 확인**: 어느 축이 위-아래?
2. **부호 확인**: +가 위인지 아래인지?
3. **Body axis 확인**: 어느 축이 앞-뒤?
4. **실험 검증**: Blender Z-up으로 변환 후 시각 확인

### 변환 디버깅

1. Blender에서 mesh 로드
2. 축 표시기(axis helper) 추가
3. 카메라 뷰(Numpad 0)에서 방향 확인
4. 등(back)이 위, 배(belly)가 아래면 정상

---

*Last updated: 2026-02-06*
