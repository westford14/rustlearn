"""Configure pytest with proper PYTHONPATH."""

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
