import time
import os
import cv2

from models.i3d.extract_i3d import ExtractI3D
from utils.utils import build_cfg_path
from omegaconf import OmegaConf
import torch
import glob
import numpy as np
import argparse
from numba import jit

#@jit
def calc_optical_flow(movie):
    output_path = os.path.split(os.path.basename(movie))[1].replace('.mp4', '') + '_flow.mp4'
    output_path = './result_video/' + output_path
    print(output_path)
    cap = cv2.VideoCapture(movie)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # VideoWriter を作成する。
    # フォーマット指定
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # 注）グレースケールの画像を出力する場合は第5引数に0を与える
    writer = cv2.VideoWriter(output_path, fmt, fps, (width, height))

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    cv2.namedWindow('frame2', cv2.WINDOW_AUTOSIZE)
    i = 0
    while (1):
        ret, frame2 = cap.read()
        if not ret:
            break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2', rgb)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        writer.write(rgb)
        prvs = next
        print(f"{i:d} / {frames:d}", end='\r')
        i += 1

    writer.write(rgb)
    cap.release()
    cv2.destroyAllWindows()

    return output_path


# torch v1.11.0 : pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
def main(args_main):
    print(torch.__version__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device_name = torch.cuda.get_device_name(0)

    # torch.cuda.set_per_process_memory_fraction(0.1, 0)
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    # less than 0.5 will be ok:
    tmp_tensor = torch.empty(int(total_memory * 0.099), dtype=torch.int8, device='cuda')
    del tmp_tensor
    torch.cuda.empty_cache()
    start_position = int(args_main.start) - 1
    end_position = int(args_main.end)

    # Select the feature type
    feature_type = 'i3d'

    # Load and patch the config
    args = OmegaConf.load(build_cfg_path(feature_type))
    args.stack_size = 12
    args.step_size = 1
    args.extraction_fps = 30

    args.streams = 'rgb'
    # args.show_pred = True
    # Load the model
    extractor1 = ExtractI3D(args)

    args.streams = 'rgb'
    # Load the model
    extractor2 = ExtractI3D(args)

    movies = glob.glob('*.mp4')
    movies.sort(reverse=False)
    movies1 = movies[start_position:end_position]
    for movie in movies1:
        flow_path = calc_optical_flow(movie)
        output_npy = os.path.basename(movie)
        output_npy = output_npy.replace('.mp4', '')

        time_sta = time.time()

        feature_dict1 = extractor1.extract(movie)
        print(f'********** {output_npy:s} rgb finished. **********')

        feature_dict_rgb = feature_dict1['rgb']
        x, y = feature_dict_rgb.shape
        if y != 1024:
            print(f'********** {output_npy:s} is ignoed. **********')
            raise ValueError("error!")
        else:
            time_end = time.time()
            tim = time_end - time_sta
            print(tim)

        feature_dict2 = extractor2.extract(flow_path)
        print(f'********** {output_npy:s} flow finished. **********')

        feature_dict_rgb = feature_dict2['rgb']
        x, y = feature_dict_rgb.shape
        if y != 1024:
            print(f'********** {output_npy:s} is ignoed. **********')
            raise ValueError("error!")
        else:
            time_end = time.time()
            tim = time_end - time_sta
            print(tim)

            feature1 = feature_dict1['rgb']
            feature2 = feature_dict2['rgb']
            concat_feature = np.concatenate([feature1, feature2], 1)
            np.save(output_npy, concat_feature)
            print(f'Name {movie:s} is Finished!!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--start', help="start position", default=1)
    parser.add_argument('--end', help="end position", default=4000)
    parser.add_argument('--iter', help="start position", default=300)
    parser.add_argument('--perplexity', help="end position", default=50)
    parser.add_argument('--neighbors', help="end position", default=300)
    args = parser.parse_args()

    os.makedirs('./result_video/', exist_ok=True)
    os.makedirs('./result_video_features/', exist_ok=True)

    feature1 = np.load('動画1.npy')

    main(args)
