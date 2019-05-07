##  The pytorch implementation of  fast-neural-style

Paper:  [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) 

data: COCO 2014 Training images dataset [80K/13GB] [(download)](http://mscoco.org/dataset/#download).

## Prerequisites

pip install requirements.txt



## Usage

 If you want to visualize the training process , you need to type `python -m visdom.server` to open visdom.

train examples:

```
CUDA_VISIBLE_DEVICES=3,5 python neural_style/neural_style.py train --dataset  ./data --style-image ./images/style-images/losses.jpg --save-model-dir ./my_models  --batch-size 8 --epochs 1 --cuda 1 --log-interval 5 --image-size 256
```

test examples:

```
python neural_style/neural_style.py eval --content-image ./images/content-images/r1.jpg  --model ./my_models/wave.pth --output-image wave_r1.jpg  --cuda 1  
```



## Result

![merge](E:\Medical image\7. Code\My project\pycharm\fast_neural_style_lifeifei\merge.jpg)