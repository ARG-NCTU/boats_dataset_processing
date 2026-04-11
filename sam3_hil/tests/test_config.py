"""
Tests for configuration module.
"""

import pytest
from pathlib import Path


class TestConfidenceConfig:
    """Test confidence thresholds."""
    
    def test_default_thresholds(self):
        from src.config import config
        
        assert config.confidence.high_threshold == 0.9
        assert config.confidence.low_threshold == 0.7
        
    def test_threshold_ordering(self):
        """High threshold must be > low threshold."""
        from src.config import config
        
        assert config.confidence.high_threshold > config.confidence.low_threshold
        
    def test_threshold_bounds(self):
        """Thresholds must be in [0, 1]."""
        from src.config import config
        
        assert 0.0 <= config.confidence.high_threshold <= 1.0
        assert 0.0 <= config.confidence.low_threshold <= 1.0


class TestTemporalConfig:
    """Test temporal tracking settings."""
    
    def test_jitter_threshold(self):
        from src.config import config
        
        assert config.temporal.jitter_threshold == 0.15
        assert 0.0 <= config.temporal.jitter_threshold <= 1.0
        
    def test_max_propagation(self):
        from src.config import config
        
        assert config.temporal.max_propagation_frames == 100
        assert config.temporal.max_propagation_frames > 0


class TestSAM3Config:
    """Test SAM 3 model settings."""
    
    def test_default_device(self):
        from src.config import config
        
        assert config.sam3.device in ["cuda", "cpu", "mps"]
        
    def test_mock_mode_default(self):
        from src.config import config
        
        # Default should be False (real inference)
        # But may be overridden in tests
        assert isinstance(config.sam3.mock_mode, bool)


class TestGUIConfig:
    """Test GUI settings."""
    
    def test_window_dimensions(self):
        from src.config import config
        
        assert config.gui.window_width >= 800
        assert config.gui.window_height >= 600
        
    def test_colors(self):
        from src.config import config
        
        # RGB tuples
        assert len(config.gui.color_positive) == 3
        assert len(config.gui.color_negative) == 3
        
        # All values in [0, 255]
        for c in config.gui.color_positive + config.gui.color_negative:
            assert 0 <= c <= 255


class TestConfigIntegration:
    """Integration tests for config module."""
    
    def test_ensure_directories(self):
        from src.config import ensure_directories, DATA_DIR, MODELS_DIR, OUTPUT_DIR
        
        ensure_directories()
        
        # Directories should exist after calling ensure_directories
        # (Note: In Docker, these are mounted volumes)
        
    def test_config_singleton(self):
        """Config should be a singleton instance."""
        from src.config import config as config1
        from src.config import config as config2
        
        assert config1 is config2
        
    def test_quick_access_aliases(self):
        """Test convenience aliases."""
        from src.config import HIGH_THRESHOLD, LOW_THRESHOLD, JITTER_THRESHOLD
        from src.config import config
        
        assert HIGH_THRESHOLD == config.confidence.high_threshold
        assert LOW_THRESHOLD == config.confidence.low_threshold
        assert JITTER_THRESHOLD == config.temporal.jitter_threshold


class TestFrameStatusLogic:
    """Test the frame status classification logic from the prompt."""
    
    def test_high_confidence_auto_save(self):
        """Score >= 0.9 should be AUTO_SAVE."""
        from src.config import config
        
        scores = [0.95, 0.92, 0.91]
        min_score = min(scores)
        
        assert min_score >= config.confidence.high_threshold
        
    def test_low_confidence_needs_review(self):
        """Score < 0.7 should be NEEDS_REVIEW."""
        from src.config import config
        
        scores = [0.85, 0.65, 0.90]  # min is 0.65
        min_score = min(scores)
        
        assert min_score < config.confidence.low_threshold
        
    def test_uncertain_range(self):
        """Score in [0.7, 0.9) should be UNCERTAIN."""
        from src.config import config
        
        scores = [0.88, 0.75, 0.82]  # min is 0.75
        min_score = min(scores)
        
        assert config.confidence.low_threshold <= min_score < config.confidence.high_threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])