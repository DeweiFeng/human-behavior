import os
import re
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
from dataset.template import BaseMultimodalDataset, MultimodalSample

class WorkStress3D(BaseMultimodalDataset):
    EMOTION_MAP = {
            'neutral': 0,
            'stress': 1
        }
    def __init__(self, data_dir):
        pass
