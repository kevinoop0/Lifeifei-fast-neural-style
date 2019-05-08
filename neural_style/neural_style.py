# -*- coding:utf-8 -*-
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import time
import torchvision as tv
import os
from tqdm import tqdm

import utils
from vgg import Vgg16
from transformer_net import TransformerNet


class Config(object):
    image_size = 256  # 图片大小
    batch_size = 8
    data_root = 'data/'  # 数据集存放路径：data/coco/a.jpg
    num_workers = 4  # 多线程加载数据
    use_gpu = True  # 使用GPU

    style_path = 'style.jpg'  # 风格图片存放路径
    lr = 1e-3  # 学习率

    env = 'neural-style'  # visdom env
    log_interval = 10  # 每10个batch可视化一次

    epoches = 2  # 训练epoch

    content_weight = 1e5  # content_loss 的权重
    style_weight = 1e10  # style_loss的权重

    model_path = None  # 预训练模型的路径
    debug_file = '/tmp/debugnn'  # touch $debug_fie 进入调试模式

    content_path = 'input.png'  # 需要进行分割迁移的图片
    result_path = 'output.png'  # 风格迁移结果的保存路径

def train(**kwargs):
    opt = Config()
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.CenterCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(opt.data_root, transform)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True)

    model = TransformerNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(),opt.lr)
    mse_loss = torch.nn.MSELoss()
    vgg = Vgg16().to(device)

    style = transform(utils.load_image(opt.style_image, size=opt.image_size))
    style = style.repeat(opt.batch_size, 1, 1, 1).to(device)
    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(x) for x in features_style]

    model.train()
    for e in range(opt.epochs):
        agg_content_loss, agg_style_loss, count = 0., 0., 0
        for step, (x, _) in tqdm(enumerate(train_loader)):
            count += len(x)
            x = x.to(device)
            y = model(x)
            features_x = vgg(utils.normalize_batch(x))
            features_y = vgg(utils.normalize_batch(y))
            # forward+loss+backward
            optimizer.zero_grad()
            content_loss = opt.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for f, g in zip(features_y, gram_style):
                gram_y = utils.gram_matrix(f)
                style_loss += mse_loss(gram_y, g)
            style_loss *= opt.style_weight
            total_loss = content_loss + style_loss

            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (step + 1) % opt.log_interval == 0:
                message = "{}\t Epoch {}:\t [{}/{}]\t content:{:.6f}\t style:{:.6f}\t total:{:.6f}".format(time.ctime(),e+1,count,len(train_loader),
                        agg_content_loss/(step+1),agg_style_loss/(step+1),(agg_style_loss+agg_content_loss)/(step+1))
                print(message)
    model.eval()
    save_model_filename = 'epoch' + str(opt.epochs) + '_' + 'batch_size' + str(opt.batch_size) + '.pth'
    save_model_path = os.path.join(opt.model_path, save_model_filename)
    torch.save(model.state_dict(), save_model_path)

def stylize(**kwargs):
    opt = Config()

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    pass
    device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')

    # 图片处理
    content_image = tv.datasets.folder.default_loader(opt.content_path)
    content_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device).detach()

    # 模型
    style_model = TransformerNet().eval()
    style_model.load_state_dict(torch.load(opt.model_path, map_location=lambda _s, _: _s))
    style_model.to(device)

    # 风格迁移与保存
    output = style_model(content_image)
    output_data = output.cpu().data[0]
    tv.utils.save_image(((output_data / 255)).clamp(min=0, max=1), opt.result_path)


if __name__ == "__main__":
    import fire
    fire.Fire()
