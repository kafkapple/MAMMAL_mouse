"""Unit tests for mammal_ext.config module."""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGPUConfig(unittest.TestCase):
    """Tests for mammal_ext.config.gpu module."""

    def test_get_default_gpu_unknown_host(self):
        """Unknown hostname should return GPU 0."""
        from mammal_ext.config.gpu import GPU_DEFAULTS, get_default_gpu
        with patch("mammal_ext.config.gpu.socket.gethostname", return_value="unknown-host"):
            self.assertEqual(get_default_gpu(), "0")

    def test_get_default_gpu_known_host(self):
        """Known hostname should return mapped GPU."""
        from mammal_ext.config.gpu import get_default_gpu
        with patch("mammal_ext.config.gpu.socket.gethostname", return_value="gpu05"):
            self.assertEqual(get_default_gpu(), "1")

    def test_configure_gpu_explicit(self):
        """Explicit gpu_id should override all defaults."""
        from mammal_ext.config.gpu import configure_gpu
        result = configure_gpu("3")
        self.assertEqual(result, "3")
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(os.environ["EGL_DEVICE_ID"], "3")
        self.assertEqual(os.environ["PYOPENGL_PLATFORM"], "egl")

    def test_configure_gpu_env_var(self):
        """GPU_ID env var should take priority over hostname."""
        from mammal_ext.config.gpu import configure_gpu
        with patch.dict(os.environ, {"GPU_ID": "5"}, clear=False):
            result = configure_gpu()
            self.assertEqual(result, "5")

    def test_configure_gpu_cuda_visible(self):
        """CUDA_VISIBLE_DEVICES should be fallback after GPU_ID."""
        from mammal_ext.config.gpu import configure_gpu
        env = {"CUDA_VISIBLE_DEVICES": "7"}
        with patch.dict(os.environ, env, clear=False):
            # Remove GPU_ID if present
            os.environ.pop("GPU_ID", None)
            result = configure_gpu()
            self.assertEqual(result, "7")


class TestLossWeights(unittest.TestCase):
    """Tests for mammal_ext.config.loss_weights module."""

    def test_defaults_no_config(self):
        """No config should return paper defaults."""
        from mammal_ext.config.loss_weights import get_loss_weights, DEFAULT_LOSS_WEIGHTS
        lw = get_loss_weights(None)
        self.assertEqual(lw.weights["theta"], DEFAULT_LOSS_WEIGHTS["theta"])
        self.assertEqual(lw.weights["bone"], DEFAULT_LOSS_WEIGHTS["bone"])
        self.assertEqual(lw.mask_step0, 0.0)
        self.assertEqual(lw.mask_step2, 3000.0)

    def test_override_single_weight(self):
        """Config override should merge with defaults."""
        from omegaconf import OmegaConf
        from mammal_ext.config.loss_weights import get_loss_weights
        cfg = OmegaConf.create({"loss_weights": {"theta": 5.0}})
        lw = get_loss_weights(cfg)
        self.assertEqual(lw.weights["theta"], 5.0)
        self.assertEqual(lw.weights["bone"], 0.5)  # Default preserved

    def test_override_mask_step(self):
        """Step-specific mask weights should be overridable."""
        from omegaconf import OmegaConf
        from mammal_ext.config.loss_weights import get_loss_weights
        cfg = OmegaConf.create({"loss_weights": {"mask_step2": 5000.0}})
        lw = get_loss_weights(cfg)
        self.assertEqual(lw.mask_step2, 5000.0)
        self.assertEqual(lw.mask_step0, 0.0)  # Default preserved

    def test_return_type(self):
        """Should return LossWeightConfig namedtuple."""
        from mammal_ext.config.loss_weights import get_loss_weights, LossWeightConfig
        lw = get_loss_weights(None)
        self.assertIsInstance(lw, LossWeightConfig)
        self.assertIsInstance(lw.weights, dict)


class TestKeypointWeights(unittest.TestCase):
    """Tests for mammal_ext.config.keypoint_weights module."""

    def test_defaults_22_keypoints(self):
        """Default should return 22 weights."""
        from mammal_ext.config.keypoint_weights import get_keypoint_weights
        kw = get_keypoint_weights(None)
        self.assertEqual(len(kw.weights), 22)
        self.assertIsInstance(kw.weights, np.ndarray)

    def test_paper_weights_applied(self):
        """Paper keypoint weights should be applied by default."""
        from mammal_ext.config.keypoint_weights import (
            get_keypoint_weights,
            MAMMAL_PAPER_KEYPOINT_WEIGHTS,
        )
        kw = get_keypoint_weights(None)
        for idx, weight in MAMMAL_PAPER_KEYPOINT_WEIGHTS.items():
            self.assertAlmostEqual(kw.weights[idx], weight,
                                   msg=f"Keypoint {idx} weight mismatch")

    def test_default_weight_is_one(self):
        """Non-special keypoints should have weight 1.0."""
        from mammal_ext.config.keypoint_weights import (
            get_keypoint_weights,
            MAMMAL_PAPER_KEYPOINT_WEIGHTS,
        )
        kw = get_keypoint_weights(None)
        for i in range(22):
            if i not in MAMMAL_PAPER_KEYPOINT_WEIGHTS:
                self.assertAlmostEqual(kw.weights[i], 1.0,
                                       msg=f"Keypoint {i} should be 1.0")

    def test_tail_step2_weight(self):
        """Tail keypoints should have configurable step2 weight."""
        from mammal_ext.config.keypoint_weights import get_keypoint_weights
        kw = get_keypoint_weights(None)
        self.assertGreater(kw.tail_step2_weight, 0)

    def test_sparse_indices_default_none(self):
        """Default config should have no sparse indices."""
        from mammal_ext.config.keypoint_weights import get_keypoint_weights
        kw = get_keypoint_weights(None)
        self.assertIsNone(kw.sparse_indices)

    def test_custom_keypoint_num(self):
        """Custom keypoint_num should change array size."""
        from mammal_ext.config.keypoint_weights import get_keypoint_weights
        kw = get_keypoint_weights(None, keypoint_num=10)
        self.assertEqual(len(kw.weights), 10)


class TestModelLoader(unittest.TestCase):
    """Tests for mammal_ext.model_loader module."""

    def test_import(self):
        """model_loader should be importable."""
        from mammal_ext.model_loader import load_body_model
        self.assertTrue(callable(load_body_model))

    def test_cache_behavior(self):
        """Cached loads should return same instance."""
        from mammal_ext import model_loader
        model_loader._body_model_cache = "test_sentinel"
        result = model_loader.load_body_model(use_cache=True)
        self.assertEqual(result, "test_sentinel")
        model_loader._body_model_cache = None  # Reset


if __name__ == "__main__":
    unittest.main()
