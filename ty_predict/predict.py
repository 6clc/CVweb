import numpy as np
import cv2
from tensorflow.python.keras.models import load_model, Model
import matplotlib.pyplot as plt

from ty_predict.utils.AdamW import AdamW
import os

data_root_dir = '/home/hanshan/Projects/ProjectsCV/AI-Web/ty_predict'

def img_pre_processing(img_path):
    x = cv2.imread(img_path)
    x = np.array(x, dtype=np.float32)
    shape = tuple(reversed(x.shape[: -1]))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)  # bgr to rgb
    x = (x - 127.5) / 255
    x = cv2.resize(x, (256, 256), interpolation=cv2.INTER_LINEAR)
    x = x[np.newaxis, ...]

    return x, shape


def sos_predict(data_dict):
    img_path = data_dict['img']
    img, _ = img_pre_processing(img_path)
    model = load_model(
        os.path.join(data_root_dir, 'checkpoints/sos.hdf5'),
        custom_objects={'AdamW': AdamW})
    return np.argmax(model.predict(img))


def sod_predict(data_dict):
    img_path = data_dict['img']
    img, shape = img_pre_processing(img_path)
    print(shape)

    subi_model = load_model(
        os.path.join(data_root_dir, 'checkpoints/sos.hdf5'),
        custom_objects={'AdamW': AdamW})
    adaptive_model = Model(inputs=subi_model.input,
                           outputs=subi_model.get_layer('sub_block5_conv3').output)

    sod_model = load_model(
        os.path.join(data_root_dir, 'checkpoints/sod.hdf5'),
        custom_objects={'AdamW': AdamW})

    adaptive = adaptive_model.predict(img)
    adaptive = adaptive.reshape((1, 1, 512, 512))
    sod_model.get_layer('adaptive_weight_layer').set_weights([adaptive])
    results = sod_model.predict(img)

    result = results[-1].squeeze()
    result = cv2.resize(result, shape)
    print(result[result > 0])
    plt.imshow(result, cmap='gray')
    plt.show()

    return results


if __name__ == '__main__':
    print(sos_predict('/home/sse/SalientObjectDetection/data/SOC6K/TrainSet/Imgs/bing_bg_1_0032.jpg'))
    print(sos_predict('/home/sse/SalientObjectDetection/data/SOC6K/TrainSet/Imgs/COCO_train2014_000000000731.jpg'))
    # sod_predict('/home/sse/SalientObjectDetection/data/SOC6K/TrainSet/Imgs/COCO_train2014_000000000731.jpg')

