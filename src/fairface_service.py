from __future__ import print_function, division

import os
from io import BytesIO

import torch
import torch.nn as nn
import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms
import dlib

RACE4_MAP = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian'}
RACE_MAP = {0: 'White', 1: 'Black', 2: 'Latino_Hispanic', 3: 'East Asian', 4: 'Southeast Asian', 5: 'Indian',6: 'Middle Eastern'}
AGE_MAP = {0: '0-2', 1: '3-9', 2: '10-19', 3: '20-29', 4: '30-39', 5: '40-49', 6: '50-59', 7: '60-69', 8: '70+'}
GENDER_MAP = {0: 'MALE', 1: 'FEMALE'}

DLIB_DETECTION_MODEL = os.environ.get('DLIB_DETECTION_MODEL','dlib_models/mmod_human_face_detector.dat')
DLIB_SHAPE_PREDICTOR_MODEL = os.environ.get('DLIB_SHAPE_PREDICTOR_MODEL','dlib_models/shape_predictor_5_face_landmarks.dat')

DEVICE_MODE = os.environ.get('DEVICE_MODE', 'cpu')
FAIR7_MODEL = os.environ.get('FAIR7_MODEL', 'fair_face_models/res34_fair_align_multi_7_20190809.pt')
FAIR4_MODEL = os.environ.get('FAIR4_MODEL', 'fair_face_models/fairface_alldata_4race_20191111.pt')


class FairFaceError(Exception):
    def __init__(self, expression='', message=''):
        self.expression = expression
        self.message = message

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def reverse_resized_rect(rect, resize_ratio):
    l = int(rect.left() / resize_ratio)
    t = int(rect.top() / resize_ratio)
    r = int(rect.right() / resize_ratio)
    b = int(rect.bottom() / resize_ratio)
    new_rect = dlib.rectangle(l, t, r, b)
    return [l, t, r, b], new_rect


class FairFace:
    def __init__(self):
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(DLIB_DETECTION_MODEL)
        self.sp = dlib.shape_predictor(DLIB_SHAPE_PREDICTOR_MODEL)
        self.device = torch.device(DEVICE_MODE)
        model_fair_7 = torchvision.models.resnet34(pretrained=True)
        model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
        #model_fair_7.load_state_dict(torch.load('fair_face_models/fairface_alldata_20191111.pt'))
        model_fair_7.load_state_dict(torch.load(FAIR7_MODEL, map_location=torch.device(DEVICE_MODE)))
        self.model_fair_7 = model_fair_7.to(self.device)
        self.model_fair_7.eval()
        model_fair_4 = torchvision.models.resnet34(pretrained=True)
        model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
        model_fair_4.load_state_dict(torch.load(FAIR4_MODEL, map_location=torch.device(DEVICE_MODE)))
        self.model_fair_4 = model_fair_4.to(self.device)
        self.model_fair_4.eval()
        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def analysis_image(self, image):
        image = self.trans(image)
        image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
        image = image.to(self.device)

        outputs = self.model_fair_7(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)
        print("fair7 ",outputs)
        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        # fair 4 class
        outputs = self.model_fair_4(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)
        print("fair4 ",outputs)
        race_outputs = outputs[:4]
        race_score4 = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        race_pred4 = np.argmax(race_score4)

        return {"gender": GENDER_MAP[gender_pred],
                "age": AGE_MAP[age_pred],
                "race": RACE_MAP[race_pred],
                "race4": RACE_MAP[race_pred4]}

    def process_image(self, content, default_max_size=800, size=300, padding=0.25):
        rects = []
        result_list = []
        img = np.array(Image.open(BytesIO(content)))
        old_height, old_width, _ = img.shape
        if old_width > old_height:
            resize_ratio = default_max_size / old_width
            new_width, new_height = default_max_size, int(old_height * resize_ratio)
        else:
            resize_ratio = default_max_size / old_height
            new_width, new_height = int(old_width * resize_ratio), default_max_size
        img = dlib.resize_image(img, cols=new_width, rows=new_height)
        dets = self.cnn_face_detector(img, 1)
        num_faces = len(dets)
        if num_faces == 0:
            raise FairFaceError("No face found")
        faces = dlib.full_object_detections()
        for detection in dets:
            rect = detection.rect
            faces.append(self.sp(img, rect))
            rect_tpl, rect_in_origin = reverse_resized_rect(rect, resize_ratio)
            rects.append(rect_tpl)
        images = dlib.get_face_chips(img, faces, size=size, padding=padding)
        for idx, image in enumerate(images):
            result = self.analysis_image(image)
            result['bbox'] = rects[idx]
            result_list.append(result)
        return result_list