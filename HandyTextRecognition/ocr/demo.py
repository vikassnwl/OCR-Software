
import os
import sys

# file_dir = os.path.dirname(main)
sys.path.append(os.getcwd())
from src import main
main.main('resize/morph/morph_ROI_q.jpg')