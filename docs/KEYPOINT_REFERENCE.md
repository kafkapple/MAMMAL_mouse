# MAMMAL Mouse 22 Keypoint Reference

**Source**: `mouse_model/keypoint22_mapper.json`

---

## Full Keypoint Mapping Table

| Index | Type | Source IDs | Joint/Vertex Names | Semantic Label |
|-------|------|------------|-------------------|----------------|
| 0 | V | [12274, 12225] | Vertex IDs | **Nose** |
| 1 | V | [4966, 5011] | Vertex IDs | **Left ear** |
| 2 | V | [13492, 13337, ...] | Vertex IDs (9 vertices) | **Right ear** |
| 3 | J | [64, 65] | neck_stretch, neck_stretch_end | **Neck (body center)** |
| 4 | V | [9043] | Vertex ID | **Body vertex** |
| 5 | J | [48, 51] | lumbar_vertebrae_0, tail_0 | **Tail base** |
| 6 | J | [54, 55] | tail_3, tail_4 | **Tail mid** |
| 7 | J | [61] | tail_9_end | **Tail tip** |
| 8 | J | [79] | fore_paw_digit_2b_l | **Left forepaw digit** |
| 9 | J | [74] | fore_paw_l | **Left forepaw** |
| 10 | J | [73] | ulna_l | **Left ulna (forearm)** |
| 11 | J | [70] | humerus_l | **Left humerus (upper arm)** |
| 12 | J | [104] | fore_paw_digit_2b_r | **Right forepaw digit** |
| 13 | J | [99] | fore_paw_r | **Right forepaw** |
| 14 | J | [98] | ulna_r | **Right ulna (forearm)** |
| 15 | J | [95] | humerus_r | **Right humerus (upper arm)** |
| 16 | J | [15] | hind_paw_digit_3c_l | **Left hindpaw digit** |
| 17 | J | [5] | hind_paw_l | **Left hindpaw** |
| 18 | J | [4] | tibia_l | **Left tibia (lower leg)** |
| 19 | J | [38] | hind_paw_digit_3c_r | **Right hindpaw digit** |
| 20 | J | [28] | hind_paw_r | **Right hindpaw** |
| 21 | J | [27] | tibia_r | **Right tibia (lower leg)** |

**Legend**:
- `V` = Vertex (mesh vertex position average)
- `J` = Joint (skeleton joint position)

---

## Keypoint Categories

### Head (idx 0-2)
| Index | Label | Description |
|-------|-------|-------------|
| 0 | Nose | Front tip of snout |
| 1 | Left ear | Left ear position |
| 2 | Right ear | Right ear position |

### Body Center (idx 3-5)
| Index | Label | Description |
|-------|-------|-------------|
| 3 | Neck | Neck/body center junction |
| 4 | Body | Central body vertex |
| 5 | Tail base | Where tail meets body (lumbar + tail_0) |

### Tail (idx 5-7)
| Index | Label | Description |
|-------|-------|-------------|
| 5 | Tail base | Tail root |
| 6 | Tail mid | Middle of tail |
| 7 | Tail tip | End of tail |

### Front Limbs - Left (idx 8-11)
| Index | Label | Description |
|-------|-------|-------------|
| 8 | Forepaw digit L | Left front paw digit |
| 9 | Forepaw L | Left front paw |
| 10 | Ulna L | Left forearm |
| 11 | Humerus L | Left upper arm |

### Front Limbs - Right (idx 12-15)
| Index | Label | Description |
|-------|-------|-------------|
| 12 | Forepaw digit R | Right front paw digit |
| 13 | Forepaw R | Right front paw |
| 14 | Ulna R | Right forearm |
| 15 | Humerus R | Right upper arm |

### Hind Limbs - Left (idx 16-18)
| Index | Label | Description |
|-------|-------|-------------|
| 16 | Hindpaw digit L | Left hind paw digit |
| 17 | Hindpaw L | Left hind paw |
| 18 | Tibia L | Left lower leg (shin) |

### Hind Limbs - Right (idx 19-21)
| Index | Label | Description |
|-------|-------|-------------|
| 19 | Hindpaw digit R | Right hind paw digit |
| 20 | Hindpaw R | Right hind paw |
| 21 | Tibia R | Right lower leg (shin) |

---

## Recommended Sparse Keypoint Configurations

### Minimal (3 keypoints) - Default Sparse Mode
```yaml
sparse_keypoint_indices: [0, 5, 3]

keypoint_weights:
  default: 0.0
  idx_0: 5.0   # Nose (head anchor)
  idx_5: 3.0   # Tail base (rear anchor)
  idx_3: 5.0   # Neck (body center)
```

**Rationale**: These 3 points span head-to-tail along the body axis, providing global pose constraint.

### Medium (5 keypoints) - Better Limb Coverage
```yaml
sparse_keypoint_indices: [0, 3, 5, 9, 13]

keypoint_weights:
  default: 0.0
  idx_0: 5.0   # Nose
  idx_3: 5.0   # Neck
  idx_5: 3.0   # Tail base
  idx_9: 3.0   # Left forepaw
  idx_13: 3.0  # Right forepaw
```

### Extended (8 keypoints) - Full Body Coverage
```yaml
sparse_keypoint_indices: [0, 3, 5, 7, 9, 13, 17, 20]

keypoint_weights:
  default: 0.0
  idx_0: 5.0   # Nose
  idx_3: 5.0   # Neck
  idx_5: 3.0   # Tail base
  idx_7: 2.0   # Tail tip
  idx_9: 3.0   # Left forepaw
  idx_13: 3.0  # Right forepaw
  idx_17: 3.0  # Left hindpaw
  idx_20: 3.0  # Right hindpaw
```

---

## Important Notes

### ⚠️ Common Mistakes to Avoid

**WRONG** (old incorrect indices):
```yaml
# DO NOT USE - These are leg joints, not body landmarks!
sparse_keypoint_indices: [0, 18, 21]  # 18=tibia_l, 21=tibia_r
```

**CORRECT** (proper body landmarks):
```yaml
sparse_keypoint_indices: [0, 5, 3]  # nose, tail_base, neck
```

### Keypoint Weight Guidelines

- **High weight (5.0)**: Critical anchor points (nose, neck)
- **Medium weight (3.0)**: Secondary anchors (tail base, paws)
- **Low weight (1.0-2.0)**: Fine detail points (digits, tail tip)
- **Zero weight (0.0)**: Unused in sparse mode

### Visualization Colors (in fitting_keypoints_compare.png)

- **Red circles**: Ground Truth (GT) keypoints
- **Green circles**: Predicted keypoints
- Overlap indicates good fitting quality
