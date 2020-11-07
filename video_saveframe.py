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
import pandas as pd

def parse_args():

    parser = argparse.ArgumentParser('Yolo v2')
    parser.add_argument('--output_dir', dest='output_dir',
                        default='output', type=str)
    parser.add_argument('--model_name', dest='model_name',
                        default=False, type=str)
    parser.add_argument('--cuda', dest='use_cuda',
                        default=False, type=bool)
    #parser.add_argument("--image_folder", type=str, default="images/setframe", help="path to dataset")
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

    images_dir = f'images/{args.save_video_name}/setframe'
    
    ##-----save video frame
    if(args.export_video_frame==True):
        vidcap = cv2.VideoCapture(args.video_path)
        def getFrame(sec,imgarr):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            hasFrames,image = vidcap.read()
            if hasFrames:
                cv2.imwrite(os.path.join(images_dir, str(count) + '.png'), image)     # save frame as png file
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
    
    
    images_names = os.listdir(images_dir)
    
    #숫자 이름대로 정렬
    #str to int
    for i in range(len(images_names)):
        images_names[i]=int(images_names[i].replace(".png","")) #testlist에서 ".png" 제거, 정수로 변환
    images_names.sort()
    
    for i in range(len(images_names)):
        images_names[i]=str(images_names[i])+".png"
        
    framenum=0
    
    list_dict={} #df
    coordinate_dict={}
    len_dict={}
    
    #testid=0 #
    
    for image_name in images_names:
        #test용: 일찍끝내기
        #---
        '''
        testid+=1
        if (testid>5):
            break;
        '''
        #---
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
        
        
        
        if (len(detections)>0):
            det_boxes = detections[:, :5].cpu().numpy()
            det_classes = detections[:, -1].long().cpu().numpy()
        else:
            det_boxes=np.array([])
            det_classes=None
        
        #---
        #df에 저장
        detect_list=[]
        coordinate_list=[]
        
        detect_list.append(cost_time)
        detect_list.append(int(1 / cost_time))
        for i in range(det_boxes.shape[0]):
            coordinate=[]
            #print(bbox[0],bbox[1],bbox[2],bbox[3])
            bbox=tuple(np.round(det_boxes[i, :4]).astype(np.int64))
            coordinate.append(bbox[0])
            coordinate.append(bbox[1])
            coordinate.append(bbox[2])
            coordinate.append(bbox[3])
            coordinate_list.append(coordinate)
            detect_list.append(classes[det_classes[i]])
        
        #print(coordinate_list)
        coordinate_dict[image_name]=str(coordinate_list) #2차원 배열 dataframe에 유지되기 위함
        #print(coordinate_dict)
        list_dict[image_name]=detect_list
        len_dict[image_name]=det_boxes.shape[0]
    
        # ---
        
        #img 저장
        im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=classes)
        plt.figure(figsize=(33,19)) #plot size
        plt.gca().xaxis.set_major_locator(NullLocator()) # delete axis
        plt.gca().yaxis.set_major_locator(NullLocator()) # delete axis
        plt.imshow(im2show)
        #plt.show()
        
        
        #save detected img
        path='/images/'
        filename = image_path.split("/")[-1].split('\\')[-1].split(".")[0]
        plt.savefig(f"images/{args.save_video_name}/detected_img/{filename}.png", bbox_inches="tight", pad_inches=0.0)
       #plt.savefig(f"images/testimg.png", bbox_inches="tight", pad_inches=0.0)
        
        imgarr=[]
     
    #딕셔너리 df 생성
    print("making dataframe...")
    res = pd.DataFrame.from_dict(list_dict, orient='index')
    countimg=list(res.columns)
    
    for i in range(0,len(countimg)):
        countimg[i]=countimg[i]-1
    
    countimg[0]='cost time'
    countimg[1]='FPS'    
    res.columns=countimg
    
    res2= pd.DataFrame.from_dict(coordinate_dict, orient='index')
    # 2차원 배열 형태로 넣고 싶음 (dataframe 안나눠지게)
    
    res2=res2.rename(columns={0:'coordinate'}) #초기 설정 0으로 되어있음
    #print(res2)
    #print(coordinate_dict)
    
    res3= pd.DataFrame.from_dict(len_dict, orient='index')
    res3=res3.rename(columns={0:'len'}) #초기 설정 0으로 되어있음
    
    res['coordinate']=None
    res['coordinate']=res2['coordinate']
    
    res['len']=None
    res['len']=res3['len']
    
    #csv 생성
    print("making csv file...")
    res.to_csv(f'csv/{args.model_name}_{args.save_video_name}.csv', mode='w')
        
    '''
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
        
        if(np.shape(pixel)!=(283, 500, 3)):
            #print("hello")
            pixel=pixel[0:283,0:500,0:3]
        

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
'''

if __name__ == '__main__':
    demo()