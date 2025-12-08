#!/usr/bin/env python3
"""Analyze body keypoint position relative to other landmarks."""

import numpy as np
import torch
import sys
sys.path.insert(0, '..')

from articulation_th import ArticulationTorch

def main():
    # Load body model
    model = ArticulationTorch()

    # Use T-pose: vertices from v_template_th, joints from compute_Tpose
    V_np = model.v_template_th.detach().cpu().numpy()
    J_np = model.compute_Tpose()[0].detach().cpu().numpy()  # [0] to get single batch

    # Keypoint positions
    mapper = model.mapper
    keypoints = {}
    for m in mapper:
        kp_id = m['keypoint']
        if kp_id < 22:
            if m['type'] == 'V':
                pos = V_np[m['ids']].mean(axis=0)
            else:
                pos = J_np[m['ids']].mean(axis=0)
            keypoints[kp_id] = pos

    # Keypoint names (DANNCE: Medial Spine = MAMMAL: body_middle)
    # See docs/keypoint_definitions.md for DANNCE vs MAMMAL comparison
    names = {
        0: 'L_ear', 1: 'R_ear', 2: 'nose',
        3: 'neck', 4: 'body_mid', 5: 'tail_root',  # Note: body_mid is NOT actual body center
        6: 'tail_mid', 7: 'tail_end',
        8: 'L_paw', 9: 'L_paw_end', 10: 'L_elbow', 11: 'L_shoulder',
        12: 'R_paw', 13: 'R_paw_end', 14: 'R_elbow', 15: 'R_shoulder',
        16: 'L_foot', 17: 'L_knee', 18: 'L_hip',
        19: 'R_foot', 20: 'R_knee', 21: 'R_hip'
    }

    print("="*60)
    print("Keypoint Positions (T-pose)")
    print("="*60)

    for kp_id in sorted(keypoints.keys()):
        pos = keypoints[kp_id]
        name = names.get(kp_id, 'unknown')
        print(f"{kp_id:2d} {name:15s} X={pos[0]:7.3f} Y={pos[1]:7.3f} Z={pos[2]:7.3f}")

    print()

    # Compare: body (4) vs neck (3) vs tail_root (5)
    body_pos = keypoints[4]
    neck_pos = keypoints[3]
    tail_root_pos = keypoints[5]

    # Calculate shoulder/hip centers
    l_shoulder = keypoints[11]
    r_shoulder = keypoints[15]
    l_hip = keypoints[18]
    r_hip = keypoints[21]

    shoulder_center = (l_shoulder + r_shoulder) / 2
    hip_center = (l_hip + r_hip) / 2
    true_body_center = (shoulder_center + hip_center) / 2

    # Y-axis is front-back in this coordinate system (nose is negative Y)
    print("="*60)
    print("Body Center Analysis (Y-axis = front-back)")
    print("="*60)
    print(f"nose (2):              Y={keypoints[2][1]:.3f}  (front)")
    print(f"neck (3):              Y={neck_pos[1]:.3f}")
    print(f"body (4):              Y={body_pos[1]:.3f}")
    print(f"tail_root (5):         Y={tail_root_pos[1]:.3f}")
    print(f"tail_end (7):          Y={keypoints[7][1]:.3f}  (back)")
    print()
    print(f"Shoulder center:       Y={shoulder_center[1]:.3f}")
    print(f"Hip center:            Y={hip_center[1]:.3f}")
    print(f"True body center:      Y={true_body_center[1]:.3f}")

    # Distance analysis (using Y-axis)
    body_to_neck = abs(body_pos[1] - neck_pos[1])
    body_to_tail = abs(body_pos[1] - tail_root_pos[1])
    body_to_true_center = abs(body_pos[1] - true_body_center[1])

    neck_to_tail = abs(neck_pos[1] - tail_root_pos[1])

    print()
    print("="*60)
    print("Y-axis Distance Analysis")
    print("="*60)
    print(f"neck to tail_root:     {neck_to_tail:.3f}")
    print(f"body to neck:          {body_to_neck:.3f}")
    print(f"body to tail_root:     {body_to_tail:.3f}")
    print(f"body to true center:   {body_to_true_center:.3f}")

    # Position ratio
    if neck_to_tail > 0:
        body_ratio = body_to_neck / neck_to_tail
        print()
        print("="*60)
        print("CONCLUSION")
        print("="*60)
        print(f"'body' keypoint is at {body_ratio:.1%} from neck toward tail_root")
        if body_ratio < 0.4:
            print("=> 'body' is CLOSER TO NECK (upper torso region)")
            print("=> NOT the true body center!")
        elif body_ratio > 0.6:
            print("=> 'body' is CLOSER TO TAIL_ROOT (lower torso region)")
        else:
            print("=> 'body' is roughly in the MIDDLE")

if __name__ == "__main__":
    main()
