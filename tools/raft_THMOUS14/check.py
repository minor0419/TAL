import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from skimage import img_as_ubyte
import copy

DEVICE = 'cuda'

def load_frame(frame):
    img = np.array(frame).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def display(img, flo):
    #img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    frame = flo / 255.0
    frame = img_as_ubyte(frame)
    #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image', frame)
    cv2.waitKey(1)
    return frame


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()

def demo(args):
    args.model = 'models/raft-things.pth'
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    start_position = int(args.start) - 1
    end_position = int(args.end) 

    with torch.no_grad():
        video_path = '/home/minoru/THUMOS14/test/*.mp4'
        video_list = glob.glob(video_path)
        video_list.sort()
        exclude_word = '_flow.mp4'
        for word in video_list:
            if word.find(exclude_word) != -1:
                video_list.remove(word)

        OK_count = 0
        video_list = video_list[start_position:end_position]
        for video_path in video_list:
            video_list_len = len(video_list)
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            expected_file_path = os.path.splitext(video_path)[0] + '_flow.mp4'
            if os.path.exists(expected_file_path):
                cap2 = cv2.VideoCapture(expected_file_path)
                width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
                height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps2 = cap2.get(cv2.CAP_PROP_FPS)
                frame_num2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if frame_num == frame_num2:
                    percent = OK_count / video_list_len * 100.0
                    print(f'{video_path:s}, OK {OK_count:d} / {video_list_len} {percent:3.2f} (%)')
                    OK_count += 1
            cap.release()
            cap2.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--start', help="start position", default=0)
    parser.add_argument('--end', help="end position", default=0)

    args = parser.parse_args()

    demo(args)
