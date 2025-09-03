# import geopandas as gpd
# import pandas as pd
# import numpy as np
#
#
# def main():
#     pass
# def import_geojson_data():
#
# if __name__ == '__main__':
#     main()



import sys
import os
import traceback

# This part is correct and adds the parent directory of 'src' to the path
# No, it adds 'src' itself. Let's correct the comment and the code.
# Best practice is to add the project root, not 'src'.
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

# Now you can import from src
from src import main_process # <-- IMPORT the module 'app' from the 'src' package

os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'

def except_hook(cls: type, exception: Exception, tb) -> None:
    """Enhanced exception handler for debugging Qt applications."""
    traceback.print_tb(tb)

# def main():
#     # sys.excepthook = except_hook
#     # Now, call the function from the imported module
#     main_process.main() # <-- CALL the function on the 'app' module

if __name__ == '__main__':
    main_process.main()