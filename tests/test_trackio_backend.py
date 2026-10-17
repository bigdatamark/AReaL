# [MODULE DOC]
# Tests for the Trackio experiment-tracking backend: TrackioConfig defaults/validation,
# the trackio field on StatsLoggerConfig, and StatsLogger init/log/finish behaviour with
# trackio, wandb, swanlab and torch.distributed patched out.
"""Tests for Trackio experiment tracking backend integration."""

from dataclasses import fields
from unittest.mock import MagicMock, patch

from areal.api.cli_args import (
    StatsLoggerConfig,
    TrackioConfig,
)


class TestTrackioConfig:
    # [CLASS DOC]
    # Tests for the TrackioConfig dataclass.
    """Tests for TrackioConfig dataclass."""

    def test_default_mode_is_disabled(self):
        # [FUNCTION DOC]
        # Asserts default mode is "disabled".
        """TrackioConfig should default to disabled mode."""
        config = TrackioConfig()
        assert config.mode == "disabled"

    def test_default_optional_fields_are_none(self):
        # [FUNCTION DOC]
        # Asserts project, name and space_id default to None.
        """Optional fields should default to None."""
        config = TrackioConfig()
        assert config.project is None
        assert config.name is None
        assert config.space_id is None

    def test_custom_values(self):
        # [FUNCTION DOC]
        # Asserts custom mode/project/name/space_id are stored.
        """TrackioConfig should accept custom values."""
        config = TrackioConfig(
            mode="online",
            project="my-project",
            name="my-run",
            space_id="user/my-space",
        )
        assert config.mode == "online"
        assert config.project == "my-project"
        assert config.name == "my-run"
        assert config.space_id == "user/my-space"

    def test_invalid_mode_raises_error(self):
        # [FUNCTION DOC]
        # Asserts an invalid mode raises ValueError "Invalid trackio mode".
        """TrackioConfig should reject invalid mode values."""
        import pytest

        with pytest.raises(ValueError, match="Invalid trackio mode"):
            TrackioConfig(mode="invalid")

    def test_all_valid_modes_accepted(self):
        # [FUNCTION DOC]
        # Asserts "disabled", "online" and "local" are accepted.
        """TrackioConfig should accept all valid mode values."""
        for mode in ("disabled", "online", "local"):
            config = TrackioConfig(mode=mode)
            assert config.mode == mode


class TestStatsLoggerConfigTrackio:
    # [CLASS DOC]
    # Tests for the trackio field on StatsLoggerConfig.
    """Tests for Trackio field in StatsLoggerConfig."""

    def test_trackio_field_exists(self):
        # [FUNCTION DOC]
        # Asserts StatsLoggerConfig declares a trackio dataclass field.
        """StatsLoggerConfig should have a trackio field."""
        field_names = [f.name for f in fields(StatsLoggerConfig)]
        assert "trackio" in field_names

    def test_trackio_field_default_is_disabled(self):
        # [FUNCTION DOC]
        # Asserts the default trackio field is a TrackioConfig in disabled mode.
        """StatsLoggerConfig.trackio should default to disabled TrackioConfig."""
        config = StatsLoggerConfig(
            experiment_name="test_exp",
            trial_name="trial_0",
            fileroot="/tmp/test",
        )
        assert isinstance(config.trackio, TrackioConfig)
        assert config.trackio.mode == "disabled"


def _make_test_config(trackio_config=None):
    # [FUNCTION DOC]
    # HELPER: Builds a BaseExperimentConfig (test_exp/trial_0, 1 epoch) with
    # stats_logger names and fileroot /tmp/test, optionally setting
    # stats_logger.trackio. Returns the config.
    """Create a minimal BaseExperimentConfig for testing StatsLogger."""
    from areal.api.cli_args import BaseExperimentConfig

    config = BaseExperimentConfig(
        experiment_name="test_exp",
        trial_name="trial_0",
        total_train_epochs=1,
    )
    config.stats_logger.experiment_name = "test_exp"
    config.stats_logger.trial_name = "trial_0"
    config.stats_logger.fileroot = "/tmp/test"
    if trackio_config is not None:
        config.stats_logger.trackio = trackio_config
    return config


def _make_ft_spec():
    # [FUNCTION DOC]
    # HELPER: Returns a MagicMock spec'd as FinetuneSpec with 1 epoch, 10 steps per
    # epoch and 10 total steps.
    """Create a mock FinetuneSpec for testing."""
    from areal.api import FinetuneSpec

    ft_spec = MagicMock(spec=FinetuneSpec)
    ft_spec.total_train_epochs = 1
    ft_spec.steps_per_epoch = 10
    ft_spec.total_train_steps = 10
    return ft_spec


class TestStatsLoggerTrackioIntegration:
    # [CLASS DOC]
    # StatsLogger trackio integration tests; each patches trackio, wandb, swanlab and
    # dist (not initialized).
    """Tests for Trackio integration in StatsLogger (mocked)."""

    @patch("areal.utils.stats_logger.trackio")
    @patch("areal.utils.stats_logger.wandb")
    @patch("areal.utils.stats_logger.swanlab")
    @patch("areal.utils.stats_logger.dist")
    def test_trackio_init_called_when_enabled(
        self, mock_dist, mock_swanlab, mock_wandb, mock_trackio
    ):
        # [FUNCTION DOC]
        # Online mode: asserts trackio.init called once and _trackio_enabled True.
        """trackio.init() should be called when mode is not disabled."""
        mock_dist.is_initialized.return_value = False

        from areal.utils.stats_logger import StatsLogger

        config = _make_test_config(TrackioConfig(mode="online"))
        logger = StatsLogger(config, _make_ft_spec())
        mock_trackio.init.assert_called_once()
        assert logger._trackio_enabled is True

    @patch("areal.utils.stats_logger.trackio")
    @patch("areal.utils.stats_logger.wandb")
    @patch("areal.utils.stats_logger.swanlab")
    @patch("areal.utils.stats_logger.dist")
    def test_trackio_not_init_when_disabled(
        self, mock_dist, mock_swanlab, mock_wandb, mock_trackio
    ):
        # [FUNCTION DOC]
        # Default (disabled): asserts trackio.init not called and _trackio_enabled
        # False.
        """trackio.init() should NOT be called when mode is disabled."""
        mock_dist.is_initialized.return_value = False

        from areal.utils.stats_logger import StatsLogger

        config = _make_test_config()  # trackio defaults to disabled
        logger = StatsLogger(config, _make_ft_spec())
        mock_trackio.init.assert_not_called()
        assert logger._trackio_enabled is False

    @patch("areal.utils.stats_logger.trackio")
    @patch("areal.utils.stats_logger.wandb")
    @patch("areal.utils.stats_logger.swanlab")
    @patch("areal.utils.stats_logger.dist")
    def test_trackio_log_called_on_commit(
        self, mock_dist, mock_swanlab, mock_wandb, mock_trackio
    ):
        # [FUNCTION DOC]
        # Online mode: asserts commit forwards the data dict to trackio.log with
        # step=global_step 0.
        """trackio.log() should be called during commit when enabled."""
        mock_dist.is_initialized.return_value = False

        from areal.utils.stats_logger import StatsLogger

        config = _make_test_config(TrackioConfig(mode="online"))
        logger = StatsLogger(config, _make_ft_spec())
        mock_trackio.log.reset_mock()

        data = {"loss/avg": 0.5, "reward/avg": 1.0}
        logger.commit(epoch=0, step=0, global_step=0, data=data)
        mock_trackio.log.assert_called_once_with(data, step=0)

    @patch("areal.utils.stats_logger.trackio")
    @patch("areal.utils.stats_logger.wandb")
    @patch("areal.utils.stats_logger.swanlab")
    @patch("areal.utils.stats_logger.dist")
    def test_trackio_finish_called_on_close(
        self, mock_dist, mock_swanlab, mock_wandb, mock_trackio
    ):
        # [FUNCTION DOC]
        # Online mode: asserts close calls trackio.finish once.
        """trackio.finish() should be called during close when enabled."""
        mock_dist.is_initialized.return_value = False

        from areal.utils.stats_logger import StatsLogger

        config = _make_test_config(TrackioConfig(mode="online"))
        logger = StatsLogger(config, _make_ft_spec())
        mock_trackio.finish.reset_mock()

        logger.close()
        mock_trackio.finish.assert_called_once()

    @patch("areal.utils.stats_logger.trackio")
    @patch("areal.utils.stats_logger.wandb")
    @patch("areal.utils.stats_logger.swanlab")
    @patch("areal.utils.stats_logger.dist")
    def test_trackio_not_logged_when_disabled(
        self, mock_dist, mock_swanlab, mock_wandb, mock_trackio
    ):
        # [FUNCTION DOC]
        # Disabled: asserts commit does not call trackio.log.
        """trackio.log() should NOT be called during commit when disabled."""
        mock_dist.is_initialized.return_value = False

        from areal.utils.stats_logger import StatsLogger

        config = _make_test_config()  # trackio defaults to disabled
        logger = StatsLogger(config, _make_ft_spec())
        mock_trackio.log.reset_mock()

        data = {"loss/avg": 0.5}
        logger.commit(epoch=0, step=0, global_step=0, data=data)
        mock_trackio.log.assert_not_called()
