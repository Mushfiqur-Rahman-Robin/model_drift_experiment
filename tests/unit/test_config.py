import pytest
import pytest
from src.config import config, Config, setup_logging
from src.api.main import startup_event
from pathlib import Path
import logging

@pytest.mark.unit
class TestConfig:
    def test_ensure_dirs(self, tmp_path):
        # Mock paths to point to a temp dir
        # We need to patch the class attributes because ensure_dirs is a classmethod
        original_model_dir = Config.MODEL_DIR
        original_data_dir = Config.DATA_DIR

        Config.MODEL_DIR = tmp_path / "models"
        Config.DATA_DIR = tmp_path / "data"

        try:
            config.ensure_dirs()
            assert Config.MODEL_DIR.exists()
            assert Config.DATA_DIR.exists()
        finally:
            # Restore
            Config.MODEL_DIR = original_model_dir
            Config.DATA_DIR = original_data_dir

    def test_setup_logging_clears_handlers(self):
        # Get the root logger
        root_logger = logging.getLogger()
        
        # Add a dummy handler to simulate pre-existing handlers
        dummy_handler = logging.NullHandler()
        root_logger.addHandler(dummy_handler)
        assert dummy_handler in root_logger.handlers

        # Call setup_logging, which should clear existing handlers
        setup_logging()

        # Assert that the dummy handler is no longer present
        assert dummy_handler not in root_logger.handlers
        # And that new handlers (stream, app.log, error.log) are present
        assert len(root_logger.handlers) == 3 # StreamHandler, FileHandler (app.log), FileHandler (error.log)


@pytest.mark.unit
async def test_startup_event():
    # This just calls ensure_dirs, so we just verify it runs without error
    # We can mock config.ensure_dirs to verify it's called if we want to be strict,
    # but running it is fine too.
    await startup_event()
