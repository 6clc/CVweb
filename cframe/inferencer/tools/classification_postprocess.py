import numpy as np

def postprocess_classification(out):
    return np.argmax(out, axis=1)