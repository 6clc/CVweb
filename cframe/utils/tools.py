import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import shutil
from matplotlib import pyplot as plt

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


