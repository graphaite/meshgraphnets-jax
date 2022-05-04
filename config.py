"""Minimal version of yacs config file for compatibility with run_model.py"""
from yacs.config import CfgNode as CN


_C = CN()
_C.SYSTEM = CN()

# Path to desired dataset
_C.DATASET_PATH = ""

# Dataset specific parameters
_C.TARGET_FIELDS = []
_C.ADD_HISTORY = None

# MLP hyperparameters
_C.NUM_LAYERS = None
_C.LATENT_SIZE = None

def get_cfg_defaults():
	return _C.clone()
