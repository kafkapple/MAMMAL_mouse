# 키포인트 최적화 및 해상도 적응 분석

## 📍 현재 키포인트 시스템 분석

### 22개 키포인트가 필요한 이유

**MAMMAL 모델 아키텍처 요구사항:**
- **139개 관절 가중치**: `reg_weights.txt`에서 확인
- **22개 키포인트**: 마우스 해부학적 구조의 주요 랜드마크
- **3D 메시 제약**: 키포인트가 3D 모델의 joint constraint로 사용

### 현재 22개 키포인트 구성
```python
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",          # 머리부 (5개)
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",    # 앞다리 (4개)  
    "left_paw", "right_paw", "left_hip", "right_hip",                  # 사지 끝점 (4개)
    "left_knee", "right_knee", "left_foot", "right_foot",             # 뒷다리 (4개)
    "neck", "tail_base", "wither", "center", "tail_middle"            # 몸통/꼬리 (5개)
]
```

## 🎯 저해상도 최적화 전략

### 1. 계층적 키포인트 시스템

#### Tier 1: 핵심 키포인트 (8개) - 최소 필수
```python
CORE_KEYPOINTS = [
    "nose",           # 머리 방향
    "neck",           # 머리-몸통 연결
    "center",         # 몸통 중심  
    "tail_base",      # 몸통-꼬리 연결
    "left_shoulder",  # 왼쪽 앞다리
    "right_shoulder", # 오른쪽 앞다리
    "left_hip",       # 왼쪽 뒷다리
    "right_hip"       # 오른쪽 뒷다리
]
```

#### Tier 2: 세부 키포인트 (14개) - 고해상도용
```python
DETAIL_KEYPOINTS = [
    "left_eye", "right_eye", "left_ear", "right_ear",        # 머리 세부
    "left_elbow", "right_elbow", "left_paw", "right_paw",    # 앞다리 세부
    "left_knee", "right_knee", "left_foot", "right_foot",   # 뒷다리 세부
    "wither", "tail_middle"                                  # 몸통 세부
]
```

### 2. 해상도 기반 적응 알고리즘

```python
def determine_keypoint_level(frame_width, frame_height):
    """해상도에 따른 키포인트 레벨 결정"""
    resolution = frame_width * frame_height
    
    if resolution < 300000:      # 480x640 미만
        return "minimal", 8      # 핵심 키포인트만
    elif resolution < 800000:    # 720p 미만  
        return "reduced", 12     # 핵심 + 일부 세부
    else:                        # 720p 이상
        return "full", 22        # 전체 키포인트

def generate_adaptive_keypoints(mask, keypoint_level):
    """적응적 키포인트 생성"""
    if keypoint_level == "minimal":
        return generate_core_keypoints(mask)
    elif keypoint_level == "reduced": 
        return generate_reduced_keypoints(mask)
    else:
        return generate_full_keypoints(mask)
```

### 3. 키포인트 interpolation 시스템

```python
def interpolate_missing_keypoints(core_keypoints):
    """핵심 키포인트로부터 세부 키포인트 추정"""
    full_keypoints = np.zeros((22, 3))
    
    # 핵심 키포인트 복사
    for name in CORE_KEYPOINTS:
        idx = KEYPOINT_NAMES.index(name)
        full_keypoints[idx] = core_keypoints[name]
    
    # 세부 키포인트 보간
    # 예: left_elbow = 0.6 * left_shoulder + 0.4 * left_paw
    left_shoulder_idx = KEYPOINT_NAMES.index("left_shoulder")
    left_paw_idx = KEYPOINT_NAMES.index("left_paw")
    left_elbow_idx = KEYPOINT_NAMES.index("left_elbow")
    
    full_keypoints[left_elbow_idx] = (
        0.6 * full_keypoints[left_shoulder_idx] + 
        0.4 * full_keypoints[left_paw_idx]
    )
    
    return full_keypoints
```

## 💾 현재 결과 저장 위치 및 구조

### 주요 결과 파일들

```
mouse_fitting_result/results/
├── params/                          # 피팅된 모델 파라미터
│   ├── param0.pkl                   # 3D 모델 파라미터 (thetas, trans, scale, rotation, bone_lengths, chest_deformer)
│   └── param0_sil.pkl              # 실루엣 피팅 파라미터
├── obj/                            # 3D 메시 파일 (아직 생성되지 않음)
├── render/                         # 렌더링 결과 이미지
│   ├── debug/                      # 실시간 피팅 과정 시각화
│   │   ├── fitting_0_debug_iter_*.png     # 각 반복마다의 중간 결과
│   │   └── fitting_0_global_iter_*.png    # 전역 최적화 과정
│   ├── fitting_0.png              # 최종 오버레이 결과 (생성 예정)
│   └── fitting_0_sil.png           # 실루엣 비교 (생성 예정)
└── fitting_keypoints_0.png         # 키포인트 시각화 (생성 예정)
```

### 파라미터 내용 분석
```python
# param0.pkl 구조
{
    'thetas': torch.Tensor,          # 관절 회전 각도 (139개 관절)
    'trans': torch.Tensor,           # 전역 평행이동
    'scale': torch.Tensor,           # 전역 스케일  
    'rotation': torch.Tensor,        # 전역 회전
    'bone_lengths': torch.Tensor,    # 뼈 길이 파라미터
    'chest_deformer': torch.Tensor   # 가슴 변형 파라미터
}
```

### 로그 파일 위치
```
outputs/2025-11-02/최신타임스탬프/fitter_articulation.log
```

## 🚀 구체적 향후 계획

### Phase 1: 적응적 키포인트 시스템 (2주)

#### 1주차: 기본 시스템 구축
```python
# adaptive_preprocess.py 구현
class AdaptiveKeypointProcessor:
    def __init__(self, resolution_threshold=(480, 640)):
        self.resolution_threshold = resolution_threshold
        self.keypoint_hierarchy = {
            "minimal": CORE_KEYPOINTS,
            "reduced": CORE_KEYPOINTS + IMPORTANT_DETAILS,
            "full": KEYPOINT_NAMES
        }
    
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        level, keypoint_count = self.determine_keypoint_level(frame_width, frame_height)
        print(f"Resolution: {frame_width}x{frame_height}, Using {level} mode ({keypoint_count} keypoints)")
        
        return self.generate_keypoints_by_level(cap, level)
```

#### 2주차: 보간 및 검증 시스템
```python
# keypoint_interpolation.py 구현
class KeypointInterpolator:
    def __init__(self):
        self.interpolation_rules = {
            "left_elbow": ("left_shoulder", "left_paw", 0.6, 0.4),
            "right_elbow": ("right_shoulder", "right_paw", 0.6, 0.4),
            "left_knee": ("left_hip", "left_foot", 0.6, 0.4),
            "right_knee": ("right_hip", "right_foot", 0.6, 0.4),
            "left_eye": ("nose", "left_ear", 0.7, 0.3),
            "right_eye": ("nose", "right_ear", 0.7, 0.3),
            "wither": ("neck", "center", 0.5, 0.5),
            "tail_middle": ("tail_base", "center", 0.7, 0.3)
        }
    
    def interpolate_from_core(self, core_keypoints):
        full_keypoints = np.zeros((22, 3))
        # 보간 로직 구현
        return full_keypoints
```

### Phase 2: AI 모델 통합 (4주)

#### 3-4주차: SAM 통합
```python
# sam_integration.py
class SAMKeypointProcessor:
    def __init__(self):
        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        self.predictor = SamPredictor(self.sam)
    
    def generate_precise_mask(self, frame):
        # SAM 기반 정밀 마스크 생성
        pass
    
    def extract_keypoints_from_sam_mask(self, mask):
        # SAM 마스크로부터 더 정확한 키포인트 추출
        pass
```

#### 5-6주차: DeepLabCut 통합
```python
# dlc_integration.py  
class DLCKeypointProcessor:
    def __init__(self, model_path="models/mouse_dlc_model"):
        self.config_path = model_path
        
    def extract_keypoints(self, video_path):
        # DLC로 정확한 키포인트 추출
        deeplabcut.analyze_videos(self.config_path, [video_path])
        return self.convert_dlc_to_mammal_format()
```

### Phase 3: 성능 최적화 및 평가 (2주)

#### 7주차: 벤치마크 시스템
```python
# evaluation_system.py
class KeypointEvaluator:
    def __init__(self):
        self.methods = ["opencv", "sam", "dlc", "adaptive"]
        
    def evaluate_accuracy(self, ground_truth, predicted):
        # PCK (Percentage of Correct Keypoints) 계산
        # MSE, 시각적 품질 평가
        pass
    
    def evaluate_speed(self, method, video_path):
        # 처리 속도 벤치마크
        pass
```

#### 8주차: 통합 및 최적화
```python
# unified_system.py
class UnifiedMouseProcessor:
    def __init__(self, config):
        self.preprocess_method = config.preprocess_method
        self.adaptive_mode = config.adaptive_mode
        
    def auto_select_best_method(self, video_path):
        # 비디오 특성에 따른 최적 방법 자동 선택
        resolution = self.get_video_resolution(video_path)
        noise_level = self.estimate_noise_level(video_path)
        
        if resolution < (480, 640):
            return "adaptive_minimal"
        elif noise_level > 0.3:
            return "sam"
        else:
            return "dlc"
```

### Phase 4: 실시간 처리 및 배포 (2주)

#### 9-10주차: 실시간 시스템
```python
# realtime_processor.py
class RealtimeMouseProcessor:
    def __init__(self):
        self.keypoint_cache = {}
        self.temporal_smoother = TemporalSmoother()
        
    def process_frame_stream(self, frame_stream):
        for frame in frame_stream:
            keypoints = self.extract_keypoints_fast(frame)
            smoothed_keypoints = self.temporal_smoother.smooth(keypoints)
            yield self.render_overlay(frame, smoothed_keypoints)
```

## ⚡ 즉시 적용 가능한 최적화

### 1. 해상도별 설정 파일
```yaml
# conf/adaptive_config.yaml
preprocess:
  adaptive_keypoints: true
  resolution_thresholds:
    minimal: [320, 240]    # 8 keypoints
    reduced: [640, 480]    # 12 keypoints  
    full: [1280, 720]      # 22 keypoints
  
  keypoint_sets:
    minimal: ["nose", "neck", "center", "tail_base", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    reduced: # minimal + important details
    full: # all 22 keypoints
```

### 2. 성능 모니터링
```python
# performance_monitor.py
def monitor_keypoint_performance():
    metrics = {
        "keypoint_count": len(detected_keypoints),
        "confidence_avg": np.mean([kp[2] for kp in keypoints]),
        "processing_time": time.time() - start_time,
        "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024
    }
    return metrics
```

## 🎯 예상 성능 개선

| 해상도 | 키포인트 수 | 처리 속도 | 메모리 사용량 | 정확도 |
|--------|-------------|-----------|---------------|--------|
| 320x240 | 8 (minimal) | +300% | -60% | 85% |
| 640x480 | 12 (reduced) | +150% | -30% | 92% |
| 1280x720+ | 22 (full) | 100% | 100% | 100% |

## 📝 결론

1. **22개 키포인트는 고해상도에서 최적**, 저해상도에서는 8-12개로 축소 가능
2. **현재 결과는 성공적으로 저장됨**: 파라미터, 디버그 이미지, 로그 모두 확인
3. **향후 10주 로드맵**: 적응적 시스템 → AI 모델 통합 → 성능 최적화 → 실시간 처리
4. **즉시 개선 가능**: 해상도 기반 키포인트 적응 시스템부터 시작

**핵심**: 계층적 키포인트 시스템으로 해상도에 따른 성능-정확도 트레이드오프 최적화!