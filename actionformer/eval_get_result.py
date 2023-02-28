# python imports
import argparse
import os
import glob
import pickle
import time
from pprint import pprint

import cv2
import numpy as np
import pandas as pd
# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed, AverageMeter, postprocess_results

import json
from PIL import ImageFont, ImageDraw, Image

def putText_japanese(img, text, point, size, color):
    # Notoフォントとする
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', size)

    # imgをndarrayからPILに変換
    img_pil = Image.fromarray(img)

    # drawインスタンス生成
    draw = ImageDraw.Draw(img_pil)

    # テキスト描画
    draw.text(point, text, fill=color, font=font)

    # PILからndarrayに変換して返す
    return np.array(img_pil)

def valid_one_epoch2(
    val_loader,
    model,

):
    print_freq = 1
    evaluator = None


    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }

    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            output = model(video_list)

            # unpack the results into ANet format
            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend(
                        [output[vid_idx]['video_id']] *
                        output[vid_idx]['segments'].shape[0]
                    )
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                  iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    return results
################################################################################
def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    # set up evaluator
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds = val_db_vars['tiou_thresholds']
        )

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    result = valid_one_epoch2(val_loader, model)
    end = time.time()
    index1 = ["video-id", "t-start", "t-end", "label", "score"]
    result_list = []

    # Set result to result list from evaluate output
    i = 0
    for i in range(len(result['t-start'])):
        result_list.append([str(result['video-id'][i]), float(result['t-start'][i]), float(result['t-end'][i]), float(result['label'][i]), float(result['score'][i])])
        i += 1

    # Select Target Movie
    i = 0
    movie_list = []
    for i in range(len(result['t-start'])):
        if result_list[i][0] == "動画3":
            apend_line = result_list[i][1:5]
            movie_list.append(apend_line)
        i += 1

    # Sort
    row_num = len(movie_list)
    movie_np = np.array(movie_list, dtype=float)
    action_num = np.argsort(movie_np[:, 0])

    action_list = np.zeros((row_num, 4), dtype=float)
    i = 0
    for action in action_num:
        action_list[i] = movie_np[action]
        i += 1

    for action in action_list:
        print(f'{action[0]:5f} {action[1]:5f} {int(action[2]):d} {action[3]*100.0:3f}')

    label_json = open('./data/next/annotations/label_define.json', 'r')
    label_json_dict = json.load(label_json)
    label_json.close()
    label = label_json_dict['label']

    cap = cv2.VideoCapture('./data/next/movies/動画3.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('./output.mp4', codec, fps, (width, height))

    frames = 0
    tt_end = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_time = 1.0 / fps * frames
        t_label = ''
        time_max = 0
        for action_item in action_list:
            t_start = action_item[0]
            t_end = action_item[1]
            t_score = action_item[3] * 100.0
            if t_start <= video_time <= t_end:
                time_length = t_end - t_start
                if time_max < time_length:
                    t_label = label[str(int(action_item[2]))]
                    time_max = time_length
                    tt_end = t_end
                    tt_score = t_score

        left_time = tt_end - video_time
        if len(t_label) != 0 and t_score > 5.0:
            text = f'Label : {t_label}'
            frame = putText_japanese(frame, text, (10, 10), 30, (0, 255, 0))
            text = f'Score : {tt_score:.3f} (%)'
            cv2.putText(frame, text=text, org=(10, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)
            text = f'Left_time : {left_time:.3f}'
            cv2.putText(frame, text=text, org=(10, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)

        cv2.imshow('frame', frame)
        video.write(frame)
        cv2.waitKey(30)
        frames += 1

    cap.release()
    video.release()
    cv2.destroyAllWindows()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    args = parser.parse_args()
    main(args)
