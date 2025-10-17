import os
import sys


project_root = os.path.dirname(__file__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import main_process


if __name__ == "__main__":
    main_process.main()
