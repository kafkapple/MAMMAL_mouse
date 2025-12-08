# Mouse 22 Keypoint Definitions

이 문서는 DANNCE 원본 키포인트 정의와 MAMMAL 프로젝트의 키포인트 정의를 비교합니다.

## DANNCE Original vs MAMMAL 키포인트 매핑

| Index | DANNCE (Original) | MAMMAL (mouse_22_defs.py) | 비고 |
|-------|-------------------|---------------------------|------|
| 0 | Left Ear | left_ear_tip | 동일 |
| 1 | Right Ear | right_ear_tip | 동일 |
| 2 | Snout | nose | 동의어 (코끝) |
| 3 | Anterior Spine | neck | 전방 척추 → 목 |
| 4 | **Medial Spine** | **body_middle** | 중간 척추 (몸 중앙 아님!) |
| 5 | Posterior Spine | tail_root | 후방 척추 → 꼬리 시작점 |
| 6 | Middle of Tail | tail_middle | 동일 |
| 7 | End of Tail | tail_end | 동일 |
| 8 | Left Hand | left_paw | 손 → 앞발 |
| 9 | - | left_paw_end | MAMMAL 추가 |
| 10 | Left Elbow | left_elbow | 동일 |
| 11 | Left Shoulder | left_shoulder | 동일 |
| 12 | Right Hand | right_paw | 손 → 앞발 |
| 13 | - | right_paw_end | MAMMAL 추가 |
| 14 | Right Elbow | right_elbow | 동일 |
| 15 | Right Shoulder | right_shoulder | 동일 |
| 16 | Left Foot | left_foot | 동일 |
| 17 | Left Knee | left_knee | 동일 |
| 18 | Left Hip | left_hip | 동일 |
| 19 | Right Foot | right_foot | 동일 |
| 20 | Right Knee | right_knee | 동일 |
| 21 | Right Hip | right_hip | 동일 |

## 주요 차이점

### 1. 척추 관련 키포인트 (Index 3, 4, 5)

DANNCE에서는 척추를 따라 3개의 키포인트를 정의:
- **Anterior Spine** (전방 척추): 목 근처
- **Medial Spine** (중간 척추): 몸통 중앙부
- **Posterior Spine** (후방 척추): 꼬리 시작점 근처

MAMMAL에서는 이를 해부학적 랜드마크로 재명명:
- **neck**: 목
- **body_middle**: 몸 중간 (주의: 실제 몸 중심이 아님!)
- **tail_root**: 꼬리 시작점

### 2. 앞발 키포인트 (Index 8, 9, 12, 13)

DANNCE: Hand (손) → MAMMAL: paw (앞발)

MAMMAL은 추가로 `paw_end` 키포인트를 정의하여 더 세밀한 앞발 추적 가능.

### 3. 용어 차이

| DANNCE 용어 | MAMMAL 용어 | 설명 |
|-------------|-------------|------|
| Snout | nose | 코끝 |
| Hand | paw | 앞발 |
| Spine | neck/body/tail_root | 해부학적 위치명 |

## 중요한 참고사항

### "body_middle" 키포인트의 실제 위치

**주의**: `body_middle` (index 4)은 이름과 달리 실제 몸 중심이 아닙니다!

분석 결과 (scripts/analyze_body_keypoint.py):
- neck (3)에서 tail_root (5) 방향으로 **62.9%** 위치
- 즉, 꼬리 쪽에 더 가�게 위치
- 실제 몸 중심 (어깨+힙 평균)과 거리가 있음

```
neck (3)             body_middle (4)        tail_root (5)
  |--------------------------|----------------------|
  0%                        62.9%                 100%
```

이는 DANNCE의 "Medial Spine" 정의를 따른 것으로, 척추의 중간 지점을 의미합니다.
실제 몸 중심이 필요한 경우, 어깨와 힙의 평균 위치를 사용하는 것이 권장됩니다.

## 관련 파일

- `mouse_22_defs.py`: 키포인트 이름 및 골격 연결 정의
- `mouse_model/keypoint22_mapper.json`: 키포인트-메시 정점 매핑
- `scripts/mesh_animation.py`: 시각화용 키포인트 레이블

## 참고 문헌

- DANNCE: https://github.com/spoonsso/dannce
- "DANNCE: Deep Animal Neural Network for Computational Ethology" (PMC8530226)
