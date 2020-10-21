# YOLOv2 코드와 YOLOv2 논문 비교
---
## YOLOv2 코드를 이용하였기 때문에 YOLOv2 논문 중 Better, Faster 부분에 대해서 설명
### [Faster](#1-Faster)
### [Better](#2-better)
---
# 1. Faster
---
- 목차  
[Darknet-19](#1-darknet-19)  
[Training for classification](#2-Training-for-classification)  
[Training for detection](#3-Training-for-detection)  
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
<br><br>

---
---
---
# 2. Better
![image](https://user-images.githubusercontent.com/45448731/95899980-46f6c900-0dcc-11eb-8f11-03ed17e406f2.png)
---
- 목차
[Multi-Scale training](#1-Multi-Scale-training)
[High Resolution Classifier](#2-High-Resolution-Classifier)
[Batch normalization]()
[Convolutional With Anchor Boxes]()
[Direct location prediction]()
[Dimension Clusters]()
[Fine-Grained Features]()


---
## 1) Multi-Scale training
- Batch를 10번 돌 때마다 320, 352, ... 608 까지 중 선택하여 이미지를 resizing
```py
# [config.py]
...
scale_step = 40  # 10씩마다 아니고 40마다로 구현
...
input_sizes = [(320, 320),
               (352, 352),
               (384, 384),
               (416, 416),
               (448, 448),
               (480, 480),
               (512, 512),
               (544, 544),
               (576, 576)] #본 코드에서는 608까지가 아니라 576까지

input_size = (416, 416)

test_input_size = (416, 416)
...
```
```py
# [train.py]
...
if cfg.multi_scale and (step + 1) % cfg.scale_step == 0:
                scale_index = np.random.randint(*cfg.scale_range) # 224보다 큰 size들에 대하여 random으로 학습
                cfg.input_size = cfg.input_sizes[scale_index]
...
```
<br><br>

---
## 2) High Resolution Classifier
- Faster의 Training for classification 마지막 항목 참조
<br><br>

---
## 3) Batch normalization
```py
# [darknet.py]
...
# Convolutional layer 만드는 함수
def conv_bn_leaky(in_channels, out_channels, kernel_size, return_module=False):
    padding = int((kernel_size - 1) / 2)
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                        stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),  # Conv layer만들 때마다 있음을 확인!
            nn.LeakyReLU(0.1, inplace=True)]
    if return_module:
        return nn.Sequential(*layers)
    else:
        return layers
...
    def _make_layers(self, layer_cfg):
        layers = []

        # set the kernel size of the first conv block = 3
        kernel_size = 3
        for v in layer_cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += conv_bn_leaky(self.in_channels, v, kernel_size) # Convolutional layer 만들 때 위의 함수 호출함
                kernel_size = 1 if kernel_size == 3 else 3
                self.in_channels = v
        return nn.Sequential(*layers)
...
```
<br><br>

---
## 4) Convolutional With Anchor Boxes
- Anchor Boxes -> detection 단계에 필요!!
- YOLOv2에서는 10 epoch동안 imagenet을 448x448로 키운 것에 대해 학습<sup>[High Resolution Classifier]()</sup>을 하지만, 실제 object detection 수행시에는 448x448 resolution이 아니라 416x416 resolution을 가지게끔 하는 것
- 본 코드에서는 [High Resolution Classifier]()는 없지만, object detection 수행 시 416x416 resolution을 가지도록 변환하는 부분 존재

```py
# [config.py]
...
test_input_size = (416, 416) # size 416x416 정의
...
```
```py
# [yolo_eval.py]
def scale_boxes(boxes, im_info):
...
    input_h, input_w = cfg.test_input_size # input image의 h,w 수정
...

def yolo_eval(yolo_output, im_info, conf_threshold=0.6, nms_threshold=0.4):
...
    # scale boxes
    boxes = scale_boxes(boxes, im_info) #yolo_eval 함수 안에서 호출
...
```
```py
# [test.py]
# test에서 detection 수행시 yolo_eval 함수 호출함으로써 input image의 size 조절
detections = yolo_eval(output, im_info, conf_threshold=args.conf_thresh,
                                       nms_threshold=args.nms_thresh)
```


- 모든 anchor box(bounding box)마다 class를 찾는다.
```py
# [yolo_eval.py]
...
# BOX 정의하는 함수
def generate_prediction_boxes(deltas_pred):
...

def yolo_eval(yolo_output, im_info, conf_threshold=0.6, nms_threshold=0.4):
...
    boxes = generate_prediction_boxes(deltas) # box 정의시 위의 함수 호출
...
# box와 함께 class 예측
boxes, conf, cls_max_conf, cls_max_id = yolo_filter_boxes(boxes, conf, classes, conf_threshold) 
...
```
<br><br>

---
## 5) Direct location prediction
![image](https://user-images.githubusercontent.com/45448731/95906311-f0da5380-0dd4-11eb-9eac-88801aba7c48.png)
- the cell is
offset from the top left corner of the image by (c_x,c_y)
- the bounding box prior has width and height p_w, p_h
```py
# [yolov2.py]
...
from loss import build_target, yolo_loss # loss.py 내부에 있는 함수 호출
...
        xy_pred = torch.sigmoid(out[:, :, 0:2]) #  σ(t_x), σ(t_y)
        conf_pred = torch.sigmoid(out[:, :, 4:5]) #  Variable of shape (B, H * W * num_anchors, 1), prediction of IoU score t_o
        hw_pred = torch.exp(out[:, :, 2:4]) # e^(tw), e^(th)
        class_score = out[:, :, 5:] # Variable of shape (B, H * W * num_anchors, num_classes), prediction of class scores (cls1, cls2 ..)
        class_pred = F.softmax(class_score, dim=-1)
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1) # Variable of shape (B, H * W * num_anchors, 4), predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)

        if training:
            output_variable = (delta_pred, conf_pred, class_score)
            output_data = [v.data for v in output_variable]
            gt_data = (gt_boxes, gt_classes, num_boxes)
            target_data = build_target(output_data, gt_data, h, w)

            target_variable = [Variable(v) for v in target_data]
            box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable) #yolo_loss 호출

            return box_loss, iou_loss, class_loss
...
```
```py
# [loss.py]
...
def yolo_loss(output, target):
...
    delta_pred_batch = output[0] # delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)
    conf_pred_batch = output[1]  # conf_pred = torch.sigmoid(out[:, :, 4:5])
    class_score_batch = output[2] # class_score = out[:, :, 5:]

    iou_target = target[0] #?<질문
    iou_mask = target[1]
    box_target = target[2]
    box_mask = target[3]
    class_target = target[4]
    class_mask = target[5]
...
    # calculate the loss, normalized by batch size.
    box_loss = 1 / b * cfg.coord_scale * F.mse_loss(delta_pred_batch * box_mask, box_target * box_mask, reduction='sum') / 2.0
    iou_loss = 1 / b * F.mse_loss(conf_pred_batch * iou_mask, iou_target * iou_mask, reduction='sum') / 2.0
    class_loss = 1 / b * cfg.class_scale * F.cross_entropy(class_score_batch_keep, class_target_keep, reduction='sum')

    return box_loss, iou_loss, class_loss
```
<br><br>

---
## 6) Dimension Clusters
- K-means clustering을 이용하여 anchor box의 size를 선택함
- 여기서는 k-means를 하여 값을 얻는 과정 포함 X
- PASCAL VOC set 기반으로 k=5로 미리 얻어진 anchors 값 사용

```py
# [config.py]
anchors = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]
...
```


<br><br>


---
## 7) Fine-Grained Features
![image](https://user-images.githubusercontent.com/45448731/95904530-9cce6f80-0dd2-11eb-885c-3175745a11b7.png)
- 없는것같다
