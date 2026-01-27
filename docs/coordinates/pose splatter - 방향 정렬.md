---
created: 2026-01-21T22:03:23
modified: 2026-01-27T14:26:08
---
# pose splatter - 방향 정렬
## pose splatter - 방향 정렬

이것은 **mean_translation을 빼지 않아서** 좌표 원점이 맞지 않는 문제입니다.

  

  viewmats는 mean_translation 기준으로 centered 되어 있는데, mesh는 원래 MAMMAL 좌표 그대로라서 offset이 생긴 것입니다.

  

  다시 분석해서 정확한 좌표 변환을 찾아야 합니다. utils.py의 auto_orient 로직을 정밀 분석하겠습니다.

  

● **Read**(src/utils.py)

  ⎿  Read **150** lines

  

● utils.py의 auto_orient 로직을 정확히 파악했습니다:

  

  # Line 88-100

  R_2 = rotation_matrix_between([0, 0, 1], up)  # Z축 → up 방향

  mean_translation = mean(R.T @ translation)     # 카메라 위치 평균

  rotation = R @ R_2.T                           # R_2.T 적용

  translation = (R @ mean_translation) + translation  # translation 조정

  positions = solve(rotation, translation)        # 카메라 위치 계산

  scale_factor = 1 / max(norm(positions))        # 정규화

  translation *= scale_factor                     # 스케일 적용

  

  **핵심 발견**: viewmats에서 mean_translation은 **빼지 않고** translation에 더해집니다. 3D 점에도 동일하게 적용해야 합니다.

  

  3D 점 변환 공식:

  P' = scale__factor * (R__2.T @ P)

  

  mean_translation은 **빼지 않습니다** - viewmats의 translation이 조정되었기 때문에 3D 점은 그대로 두어야 합니다.

  

  하지만 현재 문제는 **turntable이 90도 누웠다**는 것입니다. 이는 turntable의 world_up 설정 문제입니다.

  

  

  **utils.py의 auto_orient 변환 (Line 83-102):**

  1. up = -np.load(up_fn)["up"] - up 방향 로드 (부호 반전)

  2. R_2 = rotation_matrix_between([0,0,1], up) - Z축을 up으로 회전하는 행렬

  3. mean_translation = mean(R.T @ translation) - 카메라 위치의 평균

  4. rotation = R @ R_2.T - 카메라 회전에 R_2.T 적용

  5. translation = (R @ mean_translation) + translation - translation 조정

  6. scale_factor = 1.0 / max(norm(positions)) - 스케일 계산

  7. translation = scale_factor * translation - translation에 스케일 적용

  

  **핵심 발견:** viewmats의 **translation**에만 scale_factor가 적용되고, **world 좌표의 점들**에는 별도의 변환이 필요합니다!

  

  현재 per_frame_mesh.py에서는:

  - means @ R_2.T (회전만)

  - means * scale_factor (스케일만)

  

  하지만 **위치(translation) 보정이 빠져있습니다!**

  

  viewmats가 mean_translation을 기준으로 재정렬되었으므로, mesh 좌표도 동일하게 mean_translation을 빼줘야 합니다:

**d**(src/modules/mesh/deform/per_frame_mesh.py)

  ⎿  Read **60** lines

  

● mean_translation이 로드되었는지 확인하겠습니다:

  

● **Read**(src/modules/mesh/deform/per_frame_mesh.py)

  ⎿  Read **50** lines

  

● 문제를 발견했습니다!

  

  **Line 141**: self._mean_translation = positions.mean(axis=0)

  이건 **카메라 위치의 평균**인데, utils.py에서는 다른 방식으로 계산합니다:

  

  **utils.py Line 89**: mean_translation = np.mean(np.transpose(R, (0,2,1)) @ translation[...,None], axis=0)

  

  이건 R.T @ translation의 평균, 즉 **카메라 좌표계에서 본 원점 방향의 평균**입니다.

  

  그리고 더 중요한 건:

  - utils.py에서 translation에 mean_translation을 더하고

  - positions = solve(rotation, translation) 후에 scale_factor 계산

  

  하지만 **mesh 좌표에 적용되는 변환이 어떻게 되어야 하는지** 다시 생각해봐야 합니다.

  

  실제로 PoseSplatter의 shape_carving 모드가 어떻게 동작하는지 보겠습니다:

  

● 이제 apply_pose_transform_3d를 보겠습니다:

  

● **Read**(src/model.py)

  ⎿  Read **50** lines

  

● 이제 좌표계 흐름을 정확히 파악했습니다!

  

  **shape_carving 모드의 좌표 흐름:**

  1. Visual Hull에서 Gaussians 생성 → **VH center 기준 상대 좌표**

  2. apply_pose_transform_3d()에서 p_3d (VH center) 더함 → **절대 좌표**

  

  **mesh_per_frame 모드 (현재 문제):**

  - MAMMAL mesh → **MAMMAL 세계 좌표** (VH와 완전히 다른 좌표계!)

  - _perframe_absolute_coords=True → apply_pose_transform_3d() 스킵

  - 하지만 MAMMAL과 PoseSplatter viewmats의 좌표계가 다름!

  

  **핵심 문제:** MAMMAL mesh 좌표계와 PoseSplatter viewmats 좌표계가 **완전히 다름**


# 좌표계
  

  

**✅ 완료 요약**

  

  **문제 분석**

  

  | 실험     | init_mode      | comparison | turntable | 원인                    |

  |--------|----------------|------------|-----------|-----------------------|

  | M1     | mesh_per_frame | ❌          | ✓         | MAMMAL 좌표계(-Z up) 사용  |

  | B0, G0 | shape_carving  | ✓          | ❌         | Original 좌표계(Y-up) 사용 |

  

  **근본 원인**: init_mode에 따라 Gaussian params가 다른 좌표계에 존재하지만, turntable은 단일 좌표계만 사용했음

  

  **구현된 해결책**

  

  **1. 새 모듈:** **src/modules/core/coordinates.py**

  

  - CoordinateSystem enum: ORIGINAL, MAMMAL, OPENGL

  - get_world_up(): 좌표계별 world up 벡터 반환

  - get_coordinate_system_for_init_mode(): init_mode → 좌표계 매핑

  

  **2. turntable_renderer.py 수정**

  

  - get_turntable_cameras(coordinate_system=) 매개변수 추가

  - render_turntable_frames(coordinate_system="auto"): init_mode 자동 감지

  

  **3. train_script.py 수정**

  

  - coordinate_system="auto" 명시적 전달

  

  **좌표계 매핑**

  

  # init_mode → coordinate_system

  shape_carving    → "original" (Y-up)

  mesh_per_frame   → "mammal"   (-Z up)

  mesh_lbs         → "mammal"   (-Z up)

  keypoint         → "original" (Y-up)

  

  **문서**

  

  1. **docs/reports/260121_coordinate_system_analysis.md**: 상세 분석 보고서

  2. **docs/guides/visualization_guide.md**: 업데이트된 가이드

  3. **docs/guides/CAMERA_COORDINATE_SYSTEMS.md**: 기존 좌표계 문서

  

  **검증 결과**

  

  === CoordinateSystem Tests ===

  ORIGINAL world up: [0. 1. 0.]

  MAMMAL world up: [ 0.  0. -1.]

  

  === init__mode Detection Tests ===_

  _shape__carving        -> original

  mesh__per__frame       -> mammal

  mesh__lbs             -> mammal_

  

  _=== Turntable Camera Tests ===_

  _original   - Camera 0 up vector: [-0.34  0.94  0.  ]_

  _mammal     - Camera 0 up vector: [-0.  0. -1.]_ 


# **📊 분석 결과: 카메라 좌표계 불일치**

  

  **문제 요약: "옆으로 누워 있다" = World Up 축 차이**

  

  | 구성 요소                 | World Up                | 비고                                                 |

  |-----------------------|-------------------------|----------------------------------------------------|

  | **GT 데이터 (PoseSplatter)** | **-Z** (또는 auto_orient 후 Z) | up = -np.load(...)[\"up\"], sphere_renderer.py: -Z |

  | **Turntable Renderer**    | **+Y**                      | world_up = np.array([0, 1, 0])                     |

  

  **핵심 코드 비교**

  

  **1. Turntable Renderer (현재)** - turntable_renderer.py:191:

  world_up = np.array([0, 1, 0])  # Y-up (OpenGL 관례)

  

  **2. Sphere Renderer (데이터 기준)** - sphere_renderer.py:53:

  global_up = -np.array([0.0, 0.0, 1.0], dtype=float)  # -Z up

  

  **3. 데이터 로딩** - utils.py:75,86:

  up = -np.load(up_fn)["up"]  # 데이터에서 up 방향 로드

  # 또는

  up = np.mean(extrinsic[:, :3, 1], axis=0)  # extrinsic에서 Y축 평균

  

  ---

  **🔧 해결 방안**

  

  turntable_renderer.py의 get_turntable_cameras 함수에서 world_up을 **-Z**로 변경:

  

  # 현재 (문제)

  world_up = np.array([0, 1, 0])  # Y-up

  

  # 수정 (해결)

  world_up = np.array([0, 0, -1])  # -Z up (PoseSplatter 데이터 관례)

# Pose splatter 좌표계 오류
Pose splatter 좌표계 오류

- **Z=-0.342** (elevation 20도로 인해 XY 평면 아래에 위치)

  - **XY 평면에서 회전** (radius 0.94)

  - **Up 벡터가 -Z 방향** (약 -0.94 in Z component)

  

  

turntable 카메라는 **Y-up** 좌표계를 가정하고 있는데, 실제 Gaussian 데이터는 **다른 좌표계**를 사용하고 있어서 생쥐가 옆으로 누워 보이는 것 같습니다.

  

 comparison grid는 **실제 학습 데이터의 카메라**를 사용하고, turntable은 **가상 카메라**를 생성합니다. 문제는 가상 카메라의 좌표계가 맞지 않는 것입니다.

  

  실제 데이터의 카메라 좌표계를 확인해보겠습니다:

  

● **Search**(pattern: "viewmat|c2w|world_up", path: "src", glob: "*.py", 

  

**실제 데이터 좌표계:**

  - 카메라들의 "up" 벡터가 평균적으로 [0, 0, -1] (거의 **-Z up**)

  - 카메라들이 XY 평면 주위에 배치되어 있음

  

  **현재 turntable 좌표계:**

  - **Y-up**을 가정하고 있음 (line 187)