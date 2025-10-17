"""Command-line entry point for the tweet mapping utilities."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Run the main data processing pipeline."""
    from src.main_process import main as run_pipeline

    run_pipeline()


if __name__ == "__main__":
    main()
