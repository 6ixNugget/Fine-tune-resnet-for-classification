import numpy as np
import cv2
from scipy.misc import imsave, imread
from torchvision import transforms
import torch
from torch.autograd import Variable
from PIL import Image
import glob
import torch
import os

from finetunemodel import FineTuneResNet

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
ENLARGE_RATIO = 1.7
MODEL_PATH="./model_best.pth.tar"

def enlarge_bounding_box(x,y,w,h,ratio = ENLARGE_RATIO):
    el_w = w * ratio
    el_h = h * ratio
    x -= (el_w - w)/2
    y -= (el_h - h)/2
    return int(x), int(y), int(el_w), int(el_h)

def face_detection(bgr_img):
    """Detect and crop out faces in a BGR color image (BGR is the default for OpenCV libraries)"""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for i in range(len(faces)):
        x,y,w,h = faces[i]
        x,y,w,h = enlarge_bounding_box(x,y,w,h)
        b,g,r = cv2.split(bgr_img)
        rgb_img = cv2.merge([r,g,b])
        crop_img = rgb_img[y:y+h, x:x+w]
        imsave("cropped_"+str(i+1)+".jpg", crop_img)

def predict(model_path, input_var):
    model = FineTuneResNet("resnet50")

    model = torch.nn.DataParallel(model)
    model.cuda()

    if os.path.isfile(model_path):
        print("=> loading saved model '{}'".format(model_path))
        model_save = torch.load(model_path)
        model.load_state_dict(model_save['state_dict'])
    else:
        print("=> no saved model found at '{}'".format(model_path))
    
    return model(input_var)

def detection_demo():
    file_list = glob.glob("cropped_*.jpg")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    for filename in file_list:
        img_pil = Image.open(filename)
        img_tensor = preprocess(img_pil)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor)
        output = predict(MODEL_PATH, img_variable)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        print(filename)
        print(pred)

if __name__ == "__main__":
    bgr_img = cv2.imread('sachin.jpg')
    face_detection(bgr_img)
    detection_demo()
