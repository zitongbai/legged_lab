"""
Python module serving as a project/extension template.
"""

import os

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *

LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGGED_LAB_DATA_DIR = os.environ.get("LEGGED_LAB_DATA_DIR", os.path.join(LEGGED_LAB_ROOT_DIR, "data"))
