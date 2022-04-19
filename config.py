"""Minimal version of yacs config file for compatibility with run_model.py"""
from yacs.config import CfgNode as CN


_C = CN()
_C.SYSTEM = CN()

# Path to desired dataset
_C.DATASET_PATH = ""

def get_cfg_defaults():
	return _C.clone()
