import pytest
import pytest
from src.config import config, Config
from src.api.main import startup_event
from pathlib import Path

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

@pytest.mark.unit
async def test_startup_event():
    # This just calls ensure_dirs, so we just verify it runs without error
    # We can mock config.ensure_dirs to verify it's called if we want to be strict,
    # but running it is fine too.
    await startup_event()
