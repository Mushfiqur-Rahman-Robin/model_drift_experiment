import sys
from pathlib import Path

# Add the project root to sys.path to ensure local imports work
# and take precedence over installed packages with the same name (like 'src')
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))
