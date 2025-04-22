"""
Author: Liam Laidlaw
Filename: torch_to_onnx.py
Purpose: Converts all models in 
"""

import onnx
import os
from Predictor import *
from torch import load, save

