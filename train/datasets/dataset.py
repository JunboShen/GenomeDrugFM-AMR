"""
Compatibility wrapper.

The original implementation lives in `train/datasets/datatset.py` (typo kept for backward compatibility).
New code should import from `train/datasets/dataset.py`.
"""

from .datatset import DatasetForTasks  # noqa: F401

