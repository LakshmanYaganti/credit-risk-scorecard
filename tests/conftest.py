import sys
from pathlib import Path

# Make the repository root importable (`src`, `api`) when running pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
