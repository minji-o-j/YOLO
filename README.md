# YOLOv2 코드와 YOLOv2 논문 비교
---
## YOLOv2 코드를 이용하였기 때문에 YOLOv2 논문 중 Better, Faster 부분에 대해서 설명하도록 하겠습니다.
---
# 1. Faster
---
- 목차
[Darknet-19]()
[Training for classification]()
[Training for detection]()
---
## 1) Darknet-19
![image](https://user-images.githubusercontent.com/45448731/95813756-9351f280-0d53-11eb-8504-2a9cdfc51b9c.png)

```py
# [darknet.py]
# Conv Layer
def conv_bn_leaky(in_channels, out_channels, kernel_size, return_module=False):
    padding = int((kernel_size - 1) / 2)
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                        stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)]
    if return_module:
        return nn.Sequential(*layers)
    else:
        return layers
```
```py
# [darknet.py]
# max-pooling은 보통 2x2이용하여 resolution을 반으로 줄였지만, Global pooling은 HxW를 전부 pooling하여 1x1xC로 mapping 시킨다.
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x
```
```py
# [darknet.py]
class Darknet19(nn.Module):

   # filters 정의 부분, M: Maxpool
    cfg = {
        'layer0': [32],
        'layer1': ['M', 64],
        'layer2': ['M', 128, 64, 128],
        'layer3': ['M', 256, 128, 256],
        'layer4': ['M', 512, 256, 512, 256, 512],
        'layer5': ['M', 1024, 512, 1024, 512, 1024]
    }

    def __init__(self, num_classes=1000):
        super(Darknet19, self).__init__()
        self.in_channels = 3
       
        # self.make_layer() 함수 (하단 정의)는 파이토치의 nn.Sequential 도구로 여러 Basic Block들을 모듈 하나로 묶는 역할을 한다.
        self.layer0 = self._make_layers(self.cfg['layer0'])
        self.layer1 = self._make_layers(self.cfg['layer1'])
        self.layer2 = self._make_layers(self.cfg['layer2'])
        self.layer3 = self._make_layers(self.cfg['layer3'])
        self.layer4 = self._make_layers(self.cfg['layer4'])
        self.layer5 = self._make_layers(self.cfg['layer5'])
        
        # 정의된 부분 하단
        ##nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv = nn.Conv2d(self.in_channels, num_classes, kernel_size=1, stride=1) #ouput 채널의 개수는 num_classes=1000!
        self.avgpool = GlobalAvgPool2d() # classifier로 GAP 사용, FC 사용 하지 않은 것 확인 가능
        self.softmax = nn.Softmax(dim=1) #dim이라는 인자가 있지만, 어차리 Softmax 연산할 때 output으로 나온 모든 Tensor가 더해지면서 dim은 1차원이 된다.

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.conv(x)
        x = self.avgpool(x)
        x = self.softmax(x)

        return x

    def _make_layers(self, layer_cfg):
        layers = []

        kernel_size = 3 # 초기 시작할 때 커널 사이즈는 3
        for v in layer_cfg:
            if v == 'M': #Maxpool layer이면 커널 사이즈 2, stride 2
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)] 
            else:
                layers += conv_bn_leaky(self.in_channels, v, kernel_size)

                #Maxpool(kernel_size=2) 다음에 Conv layer kernel_size=3으로
                # kernel_size=3 다음에 Conv layer온 경우 kernel_size=1
                # kernel_size=1 다음에 또 Conv layer 온 경우 kernel_size=3
                kernel_size = 1 if kernel_size == 3 else 3 
                self.in_channels = v
        return nn.Sequential(*layers) # 이 함수가 반환한 layer 객체는 convolution 계층처럼 nn.Module로 다룰 수 있다.

    def load_weights(self, weights_file):
        weights_loader = WeightLoader()
        weights_loader.load(self, weights_file)
```
```py 
# [darknet.py]
if __name__ == '__main__':
    im = np.random.randn(1, 3, 224, 224) # 4차원 배열 생성
    im_variable = Variable(torch.from_numpy(im)).float() # Gradient 생성 (최근 version에는 `torch.autograd Variable` 지원x
    model = Darknet19()
    out = model(im_variable) #Darknet19 class 호출  #초기 im이 들어간다(Darknet 시작 부분 참고)
    print(out.size())
    print(model)
```
<br><br>

---
## 2) Training for classification

- 논문에 명시되어있는 부분
- [ ] polynomial rate decay (power of 4) #lr조절용
- [x] 160epoch
```py
# [train.py]
...
   parser.add_argument('--start_epoch', dest='start_epoch',
                           default=1, type=int)
...
    parser.add_argument('--max_epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=160, type=int) # 최대 epoch 160으로 정의
...
    # start training
    for epoch in range(args.start_epoch, args.max_epochs+1): 1~160. 160 epoch동안 반복
...
```

- [ ] learning rate(starting): 0.1
```py
# [config.py]
...
lr = 0.0001

decay_lrs = { #이후 epoch에 따라 줄어드는 부분
    60: 0.00001,
    90: 0.000001
}
...
```

- [x] random crops
- [ ] rotations
- [x] hue
- [x] saturation
- [x] exposure shifts
```py
# [augmentation.py]
...
def random_scale_translation(img, boxes, jitter=0.2):  # random crops
...
def random_distort(img, hue=.1, sat=1.5, val=1.5): # hue, saturation
...
def random_hue(img, rate=.1): # hue
...
def random_saturation(img, rate=1.5): # saturation
...
def random_exposure(img, rate=1.5): # exposure shifts
...
```

- [x] weight decay 0.0005
- [x] momentum 0.9

```py
# [config.py]
...
momentum = 0.9 # momentum
weight_decay = 0.0005 # weight decay 
...
```

- [x] stochastic gradient descent
```py
# [train.py]
...
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
...
```

- [ ] 초기에 224 224 -> 중간에 448x448 data를 learning rate 10<sup>-3</sup>에서 시작, 10 epoch 동안 fine tuning
  - multi-scale training 있지만 이 부분 없음

<br><br>

---
## 3) Training for detection
- [x]  darknet 뒤에 3x3 Conv layer
- [x] 그 뒤에 1x1 Conv layer 추가
- [ ] 마지막에 3x3x512 layer 추가  -- _conv4인가??_
```py
# [yolov2.py]
...
 # darknet backbone
        self.conv1 = nn.Sequential(darknet19.layer0, darknet19.layer1,
                                   darknet19.layer2, darknet19.layer3, darknet19.layer4)

        self.conv2 = darknet19.layer5

        # detection layers
       # darknet 뒤에 conv layer 추가
 
        self.conv3 = nn.Sequential(conv_bn_leaky(1024, 1024, kernel_size=3, return_module=True),
                                   conv_bn_leaky(1024, 1024, kernel_size=3, return_module=True))

        self.downsampler = conv_bn_leaky(512, 64, kernel_size=1, return_module=True)

        self.conv4 = nn.Sequential(conv_bn_leaky(1280, 1024, kernel_size=3, return_module=True),
                                   nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1))

...
```

- [ ] 160 epoch동안 learning rate 10<sup>-3</sup>에서 시작하여 10, 60, 90 epoch 마다 줄여나감
```py
# [config.py] #detection과 마찬가지로 이 파일에서 한번에 이용함!
...
lr = 0.0001 # 0.001이 아니라 0.0001에서 시작

decay_lrs = { #이후 10, 60, 90에서 줄이는 것이 아니라 60, 90에서만 줄임
    60: 0.00001,
    90: 0.000001
}
...
```
