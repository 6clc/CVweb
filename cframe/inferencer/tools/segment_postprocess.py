import numpy as np

def postprocess_segment(out):
    return np.argmax(out, axis=1)