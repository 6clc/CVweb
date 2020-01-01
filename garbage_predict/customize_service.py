import numpy as np
import torch
from PIL import Image
from fastai.vision import *
from torchvision.models import *
import pretrainedmodels

from fastai.vision import *
from fastai.vision.models import *
from fastai.vision.learner import model_meta
import sys

path = '/'
classes = [item for item in range(40)]
data2 = ImageDataBunch.single_from_classes(
    path, classes, size=224, resize_method=ResizeMethod.SQUISH).normalize(imagenet_stats)


def se_resnext50(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
    return model
# PTServingBaseService
class garbage_classify_service():
    def __init__(self, model_name=None, model_path=None):
        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = '/home/hanshan/Projects/ProjectsCV/AI-Web/garbage_predict/se-resnext50.pth'
        self.signature_key = 'predict_images'

        # self.input_size = 224  # the input image size of the model

        # add the input and output key of your pb model here,
        # these keys are defined when you save a pb file
        self.input_key_1 = 'input_img'
        self.output_key_1 = 'output_score'

        self.learn = create_cnn(data2, se_resnext50, pretrained=False, cut=-2)
        self.learn.load(self.model_path.replace('.pth', ''))

        self.label_id_name_dict = {
                "0": "其他垃圾/一次性快餐盒",
                "1": "其他垃圾/污损塑料",
                "2": "其他垃圾/烟蒂",
                "3": "其他垃圾/牙签",
                "4": "其他垃圾/破碎花盆及碟碗",
                "5": "其他垃圾/竹筷",
                "6": "厨余垃圾/剩饭剩菜",
                "7": "厨余垃圾/大骨头",
                "8": "厨余垃圾/水果果皮",
                "9": "厨余垃圾/水果果肉",
                "10": "厨余垃圾/茶叶渣",
                "11": "厨余垃圾/菜叶菜根",
                "12": "厨余垃圾/蛋壳",
                "13": "厨余垃圾/鱼骨",
                "14": "可回收物/充电宝",
                "15": "可回收物/包",
                "16": "可回收物/化妆品瓶",
                "17": "可回收物/塑料玩具",
                "18": "可回收物/塑料碗盆",
                "19": "可回收物/塑料衣架",
                "20": "可回收物/快递纸袋",
                "21": "可回收物/插头电线",
                "22": "可回收物/旧衣服",
                "23": "可回收物/易拉罐",
                "24": "可回收物/枕头",
                "25": "可回收物/毛绒玩具",
                "26": "可回收物/洗发水瓶",
                "27": "可回收物/玻璃杯",
                "28": "可回收物/皮鞋",
                "29": "可回收物/砧板",
                "30": "可回收物/纸板箱",
                "31": "可回收物/调料瓶",
                "32": "可回收物/酒瓶",
                "33": "可回收物/金属食品罐",
                "34": "可回收物/锅",
                "35": "可回收物/食用油桶",
                "36": "可回收物/饮料瓶",
                "37": "有害垃圾/干电池",
                "38": "有害垃圾/软膏",
                "39": "有害垃圾/过期药物"
            }



    def _preprocess(self, data):
        # 预处理成{key:input_batch_var}，input_batch_var为模型输入张量
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = open_image(file_content)
            preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, img):
        # img = data
        pred_class, pred_idx, outputs = self.learn.predict(img)
        pred_label = int(pred_class)
        result = {'result': self.label_id_name_dict[str(pred_label)]}
        return result


def garbage_predict(data_dict):
    img_path = data_dict['img']
    img = open_image(img_path)
    tester = garbage_classify_service()
    return tester._inference(img)['result']

if __name__ == '__main__':
    data_dict = dict(
        img='/home/hanshan/Projects/ProjectsCV/AI-Web/static/garbage/index.png')
    print(garbage_predict(data_dict))