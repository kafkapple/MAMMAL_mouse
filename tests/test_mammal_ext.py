"""Tests for mammal_ext package."""

import pytest
import os
import numpy as np


class TestGPUConfig:
    """Tests for GPU configuration module."""

    def test_get_default_gpu(self):
        """Test default GPU detection."""
        from mammal_ext.config.gpu import get_default_gpu, GPU_DEFAULTS

        default = get_default_gpu()
        assert isinstance(default, str)
        assert default in ['0', '1', '2', '3']  # Valid GPU IDs

    def test_configure_gpu(self):
        """Test GPU configuration."""
        from mammal_ext.config.gpu import configure_gpu

        # Test explicit GPU ID
        gpu_id = configure_gpu('0')
        assert gpu_id == '0'
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
        assert os.environ['EGL_DEVICE_ID'] == '0'
        assert os.environ['PYOPENGL_PLATFORM'] == 'egl'

    def test_gpu_defaults_dict(self):
        """Test GPU defaults dictionary structure."""
        from mammal_ext.config.gpu import GPU_DEFAULTS

        assert isinstance(GPU_DEFAULTS, dict)
        assert 'gpu05' in GPU_DEFAULTS
        assert GPU_DEFAULTS['gpu05'] == '1'


class TestLossWeights:
    """Tests for loss weight configuration module."""

    def test_get_loss_weights_defaults(self):
        """Test default loss weights."""
        from mammal_ext.config.loss_weights import get_loss_weights, DEFAULT_LOSS_WEIGHTS

        lw = get_loss_weights()
        assert lw.weights == DEFAULT_LOSS_WEIGHTS
        assert lw.mask_step0 == 0.0
        assert lw.mask_step1 == 0.0
        assert lw.mask_step2 == 3000.0

    def test_get_loss_weights_with_config(self):
        """Test loss weights from config."""
        from mammal_ext.config.loss_weights import get_loss_weights
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            'loss_weights': {
                'theta': 5.0,
                '2d': 0.5,
                'mask_step2': 5000.0,
            }
        })

        lw = get_loss_weights(cfg)
        assert lw.weights['theta'] == 5.0
        assert lw.weights['2d'] == 0.5
        assert lw.weights['bone'] == 0.5  # Default preserved
        assert lw.mask_step2 == 5000.0

    def test_apply_silhouette_mode_weights(self):
        """Test silhouette mode weight adjustments."""
        from mammal_ext.config.loss_weights import apply_silhouette_mode_weights

        weights = {'2d': 0.2, 'theta': 3.0, 'bone': 0.5, 'scale': 0.5}
        result = apply_silhouette_mode_weights(weights)

        assert result['2d'] == 0
        assert result['theta'] == 10.0
        assert result['bone'] == 2.0
        assert result['scale'] == 50.0


class TestKeypointWeights:
    """Tests for keypoint weight configuration module."""

    def test_get_keypoint_weights_defaults(self):
        """Test default keypoint weights."""
        from mammal_ext.config.keypoint_weights import get_keypoint_weights, MAMMAL_PAPER_KEYPOINT_WEIGHTS

        kw = get_keypoint_weights()
        assert len(kw.weights) == 22
        assert kw.weights[4] == MAMMAL_PAPER_KEYPOINT_WEIGHTS[4]  # 0.4
        assert kw.weights[5] == MAMMAL_PAPER_KEYPOINT_WEIGHTS[5]  # 2.0
        assert kw.tail_step2_weight == 10.0

    def test_get_keypoint_weights_with_config(self):
        """Test keypoint weights from config."""
        from mammal_ext.config.keypoint_weights import get_keypoint_weights
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            'fitter': {'keypoint_num': 22},
            'keypoint_weights': {
                'default': 1.0,
                'idx_5': 3.0,
                'tail_step2': 15.0,
            }
        })

        kw = get_keypoint_weights(cfg)
        assert kw.weights[5] == 3.0
        assert kw.weights[0] == 1.0  # Default
        assert kw.tail_step2_weight == 15.0

    def test_sparse_keypoint_mode(self):
        """Test sparse keypoint mode."""
        from mammal_ext.config.keypoint_weights import get_keypoint_weights
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            'fitter': {
                'keypoint_num': 22,
                'sparse_keypoint_indices': [0, 5, 18],
            },
            'keypoint_weights': {
                'default': 0.0,
            }
        })

        kw = get_keypoint_weights(cfg)
        assert kw.weights[0] == 1.0  # Sparse index
        assert kw.weights[5] == 1.0  # Sparse index
        assert kw.weights[18] == 1.0  # Sparse index
        assert kw.weights[1] == 0.0  # Not in sparse indices
        assert kw.sparse_indices == [0, 5, 18]

    def test_apply_step2_tail_weights(self):
        """Test step2 tail weight application."""
        from mammal_ext.config.keypoint_weights import apply_step2_tail_weights, TAIL_KEYPOINT_INDICES

        weights = np.ones(22)
        result = apply_step2_tail_weights(weights, tail_weight=10.0)

        for idx in TAIL_KEYPOINT_INDICES:
            assert result[idx] == 10.0
        assert result[0] == 1.0  # Non-tail keypoint unchanged


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
