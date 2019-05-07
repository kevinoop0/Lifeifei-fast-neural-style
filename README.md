##  The simple pytorch implementation of  fast-neural-style

Paper:  [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) 

train dataset : COCO 2014 Training images dataset [80K/13GB] [(download)](http://mscoco.org/dataset/#download).



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

![merge](http://a2.qpic.cn/psb?/V12kySKV4IhBFe/V8Y*rluA3lrVHUbeb9GMTT9Km9vBZa7Uv95.oebJXsM!/b/dEkBAAAAAAAA&ek=1&kp=1&pt=0&bo=OAQ4BAAAAAARNwA!&tl=3&vuin=1577159875&tm=1557223200&sce=60-2-2&rf=viewer_4)