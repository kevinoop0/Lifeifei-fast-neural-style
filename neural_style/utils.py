# -*- coding:utf-8 -*-
from PIL import Image

def load_image(file_name,size=None,scale=None):
    img = Image.open(file_name)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

def save_image(file_name,data):  # tensor->numpy->PIL
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')  # 四维数据接收三个参数tensor:C*H*W  PIL:H*W*C
    img = Image.fromarray(img)
    img.save(file_name)

def gram_matrix(y):
    (b, c, h, w) = y.size()
    feature = y.view(b, c, h * w)
    feature_t = feature.transpose(1, 2)
    gram = feature.bmm(feature_t) / (c * h * w)  # bmm ->batch mul
    return gram  # size b*c*c

def normalize_batch(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch-mean)/std
