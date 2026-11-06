"""``python3 -m trainit_data`` = the ``trainit-data`` CLI (useful where console scripts are not on PATH)."""
import sys

from .cli import main

sys.exit(main())
