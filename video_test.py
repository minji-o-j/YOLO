import os
import argparse
import time
import torch
from torch.autograd import Variable
from PIL import Image
from test import prepare_im_data
from yolov2 import Yolov2
from yolo_eval import yolo_eval
from util.visualize import draw_detection_boxes
import matplotlib.pyplot as plt
from util.network import WeightLoader
import cv2
from matplotlib.ticker import NullLocator
import numpy as np


def parse_args():

    parser = argparse.ArgumentParser('Yolo v2')
    parser.add_argument('--output_dir', dest='output_dir',
                        default='output', type=str)
    parser.add_argument('--model_name', dest='model_name',
                        default=False, type=str)
    parser.add_argument('--cuda', dest='use_cuda',
                        default=False, type=bool)
    parser.add_argument("--image_folder", type=str, default="images/setframe", help="path to dataset")
    parser.add_argument("--video_path", type=str, default=False, help="video to detect")
    parser.add_argument("--export_video_frame",default=False,type=bool) #비디오 프레임부터 꺼내야할때
    parser.add_argument("--save_video_name",default=False,type=str) #비디오 이름 지정
    args = parser.parse_args()
    return args


def demo():
    args = parse_args()
    print('call with args: {}'.format(args))

    # set model

    classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')

    model = Yolov2()
    #weight_loader = WeightLoader()
    #weight_loader.load(model, 'output/'+args.model_name+'.pth')#'yolo-voc.weights')
    
    model_path = os.path.join(args.output_dir, args.model_name+'.pth')
    print('loading model from {}'.format(model_path))
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    
    if args.use_cuda:
        model.cuda()

    model.eval()
    print('model loaded')

    ##-----save video frame
    if(args.export_video_frame==True):
        vidcap = cv2.VideoCapture(args.video_path)
        def getFrame(sec,imgarr):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            hasFrames,image = vidcap.read()
            if hasFrames:
                cv2.imwrite(os.path.join(args.image_folder, str(count) + '.png'), image)     # save frame as png file
            return hasFrames

        sec = 0
        frameRate = 0.042  #it will capture image in each 0.042 second (**24frame정도?)
        count=1
        imgarr=[]
        success = getFrame(sec,imgarr)

        num=0
        while success:
            if num%100==0:
                print(num)
            num+=1
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 2)
            success = getFrame(sec,imgarr)

        print("frame: "+str(num))
    ##-----
    
    #image
    images_dir = 'images/setframe'
    images_names = os.listdir(images_dir)
    
    framenum=0
    for image_name in images_names:
        image_path = os.path.join(images_dir, image_name)
        img = Image.open(image_path)
        im_data, im_info = prepare_im_data(img)

        if args.use_cuda:
            im_data_variable = Variable(im_data).cuda()
        else:
            im_data_variable = Variable(im_data)

        tic = time.time()

        yolo_output = model(im_data_variable)
        yolo_output = [item[0].data for item in yolo_output]
        detections = yolo_eval(yolo_output, im_info, conf_threshold=0.6, nms_threshold=0.4)

        toc = time.time()
        cost_time = toc - tic
        framenum+=1
        print(framenum,': im detect, cost time {:4f}, FPS: {}'.format(
            toc-tic, int(1 / cost_time)))
        
        
        # edit -> 감지x시
        #print(detections)
        if (len(detections)>0):
            det_boxes = detections[:, :5].cpu().numpy()
            det_classes = detections[:, -1].long().cpu().numpy()
        else:
            det_boxes=np.array([])
            det_classes=None
            
        im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=classes)
        plt.figure(figsize=(33,19)) #plot size
        plt.gca().xaxis.set_major_locator(NullLocator()) # delete axis
        plt.gca().yaxis.set_major_locator(NullLocator()) # delete axis
        plt.imshow(im2show)
        #plt.show()
        
        
        #save detected img
        path='/images/'
        filename = image_path.split("/")[-1].split('\\')[-1].split(".")[0]
        plt.savefig(f"images/detected_img/{filename}.png", bbox_inches="tight", pad_inches=0.0)
       #plt.savefig(f"images/testimg.png", bbox_inches="tight", pad_inches=0.0)
        
        imgarr=[]
    
    path_dir = 'images/detected_img/'
    file_list = os.listdir(path_dir)
    
    #숫자 이름대로 정렬
    #str to int
    for i in range(len(file_list)):
        file_list[i]=int(file_list[i].replace(".png","")) #testlist에서 ".png" 제거, 정수로 변환
    file_list.sort()
    
    for i in range(len(file_list)):
        file_list[i]=str(file_list[i])+".png"
    
    #print(file_list) #숫자 순서대로 정렬된 것 확인함
    
    for png in file_list:
        #print(png)
        #image = Image.open(path_dir + png).convert("RGB")
        image=cv2.imread(path_dir + png)
        #print(image)
        pixel = np.array(image)
        #print(np.shape(pixel))
        #pixel2=np.delete(pixel, 3, axis = 2)
        print(np.shape(pixel))
        if(np.shape(pixel)!=(1443, 2562, 3)):
            #print("hello")
            pixel=pixel[0:1443,0:2562,0:3]
        '''
        if(np.shape(pixel)!=(283, 500, 3)):
            #print("hello")
            pixel=pixel[0:283,0:500,0:3]
        '''

        #print(np.shape(pixel2))
        imgarr.append(pixel)
      
    #print(np.shape(imgarr))

        

    fps = 24 #24 #frame per second
    
    pathOut = f'images/{args.model_name}_{args.save_video_name}_fixsize.mp4'
    size=(2562,1443)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(imgarr)):
        # writing to a image array
        out.write(imgarr[i])
        #print(imgarr[i])
    
    out.release()
    
if __name__ == '__main__':
    demo()
