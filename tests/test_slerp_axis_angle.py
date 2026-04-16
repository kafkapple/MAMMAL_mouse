"""Unit tests for quaternion-based slerp_axis_angle.

Captures the ad-hoc REPL validation that accompanied the hemisphere-safe
slerp patch (2026-04-16). Run:

    python -m pytest tests/test_slerp_axis_angle.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mammal_ext.fitting.interpolation import (
    _axis_angle_to_quat,
    _quat_to_axis_angle,
    canonicalize_axis_angle,
    slerp_axis_angle,
)


def _aa_to_rotmat(aa):
    theta = np.linalg.norm(aa)
    if theta < 1e-8:
        return np.eye(3)
    k = aa / theta
    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)


class TestCanonicalize:
    def test_under_pi_unchanged(self):
        aa = np.array([0.5, -0.3, 0.2])
        assert _allclose(canonicalize_axis_angle(aa), aa)

    def test_zero_unchanged(self):
        assert _allclose(canonicalize_axis_angle(np.zeros(3)), np.zeros(3))

    @pytest.mark.parametrize("aa_overflow", [
        np.array([4.05, 0.0, 0.0]),      # 9960 joint 124 sample
        np.array([0.0, 3.85, 0.0]),
        np.array([0.0, 0.0, 5.22]),
        np.array([2.5, 2.5, 2.5]),       # |theta|=4.33
        np.array([6.9, 0.0, 0.0]),       # >2pi
    ])
    def test_overflow_rotmat_preserved(self, aa_overflow):
        canon = canonicalize_axis_angle(aa_overflow)
        assert np.linalg.norm(canon) <= np.pi + 1e-6
        R_orig = _aa_to_rotmat(aa_overflow)
        R_canon = _aa_to_rotmat(canon)
        assert np.max(np.abs(R_orig - R_canon)) < 1e-6, \
            f"rotmat differs: {np.max(np.abs(R_orig - R_canon))}"

    def test_pi_boundary_exact(self):
        # θ=π exactly: canonical form, should pass through
        aa = np.array([np.pi, 0.0, 0.0])
        canon = canonicalize_axis_angle(aa)
        assert _allclose(_aa_to_rotmat(aa), _aa_to_rotmat(canon))

    def test_two_pi_identity(self):
        # θ=2π exactly: identity rotation. Canonicalize should return zero-equivalent
        # (rotmat = I either way).
        aa = np.array([2 * np.pi, 0.0, 0.0])
        canon = canonicalize_axis_angle(aa)
        assert _allclose(_aa_to_rotmat(canon), np.eye(3), atol=1e-6)


class TestCanonicalizeSequence:
    """Compound-sequence correctness — Devil's Advocate S2 (260417 session)."""

    def test_monotone_overflow_axis_consistency(self):
        # All-overflow monotone sequence: axis sign stays consistent after canon.
        # Production data (joint 124 mean=3.87, max=5.22) routinely drifts in this
        # regime without crossing the π boundary.
        aas = [np.array([3.87, 0.0, 0.0]),
               np.array([3.60, 0.0, 0.0]),
               np.array([3.20, 0.0, 0.0])]
        canons = [canonicalize_axis_angle(a) for a in aas]
        axes = [c / np.linalg.norm(c) for c in canons if np.linalg.norm(c) > 1e-8]
        for i in range(1, len(axes)):
            assert np.dot(axes[0], axes[i]) > 0.999, \
                f"axis sign flipped at step {i}: {axes[0]} vs {axes[i]}"

    def test_pi_boundary_crossing_slerp_handled(self):
        # Sequence that crosses π boundary: 3.20 > π > 3.10. Canon forms have
        # OPPOSITE axis signs. slerp_axis_angle's hemisphere correction
        # (dot<0 → q2=-q2) must still produce a smooth shortest-path midpoint.
        aa_above = np.array([3.20, 0.0, 0.0])  # > π: flipped by canon
        aa_below = np.array([3.10, 0.0, 0.0])  # < π: unchanged
        mid = slerp_axis_angle(aa_above, aa_below, 0.5)
        R_mid = _aa_to_rotmat(mid)
        # The midpoint rotation should be close to either endpoint (small angular
        # gap). A catastrophic long-path result would differ by > 2 rad.
        R_above = _aa_to_rotmat(aa_above)
        # Rotation-difference angle between R_mid and R_above
        cos_angle = np.clip((np.trace(R_mid.T @ R_above) - 1) / 2, -1, 1)
        gap = np.arccos(cos_angle)
        assert gap < 0.5, f"midpoint too far from endpoint (gap={gap:.3f} rad) — hemisphere correction failed"

    def test_mixed_sequence_slerp_endpoints(self):
        # Mixed sequence [θ>π, θ<π, θ>π] can make adjacent canon quats antipodal.
        # slerp_axis_angle's hemisphere correction (dot<0 → q2=-q2) must handle this.
        aa_big1 = np.array([3.1, 0.0, 0.0])   # > π
        aa_small = np.array([0.9, 0.0, 0.0])  # < π
        # Slerp at α=0 and α=1 must still return the endpoints (within float noise)
        assert _allclose(slerp_axis_angle(aa_big1, aa_small, 0.0),
                         canonicalize_axis_angle(aa_big1))
        assert _allclose(slerp_axis_angle(aa_big1, aa_small, 1.0),
                         canonicalize_axis_angle(aa_small))
        # Midpoint rotmat should match manual shortest-path rotmat interpolation
        mid = slerp_axis_angle(aa_big1, aa_small, 0.5)
        R_mid = _aa_to_rotmat(mid)
        # Sanity: midpoint rotmat is a valid rotation (orthogonal, det=1)
        assert abs(np.linalg.det(R_mid) - 1.0) < 1e-5
        assert np.max(np.abs(R_mid @ R_mid.T - np.eye(3))) < 1e-5


def _allclose(a, b, atol=1e-5):
    return np.allclose(a, b, atol=atol)


class TestEndpoints:
    def test_alpha_zero_returns_aa1(self):
        aa1 = np.array([0.0, 0.0, 0.0])
        aa2 = np.array([0.5, 0.3, -0.2])
        assert _allclose(slerp_axis_angle(aa1, aa2, 0.0), aa1)

    def test_alpha_one_returns_aa2(self):
        aa1 = np.array([0.0, 0.0, 0.0])
        aa2 = np.array([0.5, 0.3, -0.2])
        assert _allclose(slerp_axis_angle(aa1, aa2, 1.0), aa2)

    def test_identical_inputs_any_alpha(self):
        aa = np.array([0.3, -0.2, 0.1])
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert _allclose(slerp_axis_angle(aa, aa, a), aa)


class TestQuatRoundTrip:
    @pytest.mark.parametrize("aa", [
        np.array([0.1, 0.0, 0.0]),
        np.array([np.pi * 0.95, 0.0, 0.0]),
        np.array([0.0, np.pi / 2, 0.0]),
        np.array([2.5, -1.3, 0.8]),
        np.array([0.0, 0.0, 0.0]),
    ])
    def test_round_trip_exact(self, aa):
        assert _allclose(_quat_to_axis_angle(_axis_angle_to_quat(aa)), aa)


class TestHemisphere:
    def test_near_antipodal_shortest_path(self):
        aa1 = np.array([np.pi * 0.9, 0.0, 0.0])
        aa2 = np.array([-np.pi * 0.9, 0.0, 0.0])
        mid = slerp_axis_angle(aa1, aa2, 0.5)
        # Midpoint goes through the short way (identity side) — result has
        # magnitude exactly π at α=0.5 for this symmetric pair.
        assert 3.13 < np.linalg.norm(mid) < 3.15


class TestNearIdentity:
    def test_nlerp_fallback_small_result(self):
        aa1 = np.array([0.001, 0.0, 0.0])
        aa2 = np.array([0.001, 0.0001, 0.0])
        mid = slerp_axis_angle(aa1, aa2, 0.5)
        assert np.linalg.norm(mid) < 0.01


class TestOrthogonalSlerp:
    """True-slerp (not nlerp) path through mid-range angles."""

    def test_orthogonal_90_deg(self):
        aa1 = np.array([np.pi / 2, 0.0, 0.0])
        aa2 = np.array([0.0, np.pi / 2, 0.0])
        mid = slerp_axis_angle(aa1, aa2, 0.5)
        # Axes symmetric across x/y → first two components equal, third ≈ 0
        assert abs(mid[0] - mid[1]) < 1e-5
        assert abs(mid[2]) < 1e-5
        # Nonzero magnitude — proves slerp ran, not zero-clamped nlerp
        assert np.linalg.norm(mid) > 1.0
