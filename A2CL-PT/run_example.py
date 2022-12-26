import os
import numpy as np
import torch
import json
from model import Model
from detection import postprocess_filter, detect_segments, postprocess_ambiguous_v
from datetime import timedelta
import cv2


def sort_key(x):
    return x[1]


def sort_value(x):
    return x[1]


# example presented in the examples folder (LongJump.mp4)
datapath = './dataset/THUMOS14'
vname = 'video_test_0001369'
#vname = 'video_test_0001380'
idx_end = 360
# weight = './weights/best2.pt'
weight = './output/THUMOS14_200_12-19_17-43-48_1/THUMOS14_200_12-19_17-43-48_1_006000.pt'

# load data
feature = np.load(os.path.join(datapath, 'feature_val.npy'), allow_pickle=True)[()][vname][:idx_end]
dict_annts = json.load(open(os.path.join(datapath, 'annotation.json'), 'r'))
cnames = dict_annts['list_classes']
set_ambiguous = dict_annts['set_ambiguous']
fps_extracted = dict_annts['miscellany']['fps_extracted']
len_feature_chunk = dict_annts['miscellany']['len_feature_chunk']

# build model
model = Model(len(cnames), 40, 0.6)
model.load_state_dict(torch.load(weight))
model.eval()

# process
preds_cwize_list = []
tcam = model(torch.from_numpy(np.array([feature])))[-1].data.numpy()
predictions, confidences = postprocess_filter(tcam, 8, 0.4)
preds_cwise = detect_segments(predictions, confidences, cnames, 0, 0)
preds_cwise = postprocess_ambiguous_v(preds_cwise, set_ambiguous, vname, fps_extracted / len_feature_chunk)

# print in the order of confidence
print('Detection results of %s:' % vname)
i = 1
action_list = []
for c in preds_cwise:
    for p in preds_cwise[c]:
        s, e = p[1] * len_feature_chunk / fps_extracted, p[2] * len_feature_chunk / fps_extracted
        prob = p[3]
        action_list.append([c, s, e, prob])
        print('%3d) %s, %5.1f ~ %5.1f (sec) %3.1f' % (i, c, s, e, prob))
        i += 1
print('-----------------------------------')
action_list.sort(key=sort_key)
i = 0
end_max = 0
for action in action_list:
    c = action[0]
    s = action[1]
    e = action[2]
    p = action[3]
    print('%3d) %s, %5.1f ~ %5.1f (sec) %3.1f' % (i, c, s, e, p))
    i += 1
    if end_max <= e:
        end_max = e


cap = cv2.VideoCapture('./examples/video_test_0001369.mp4')

# 動画のプロパティを取得
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width2 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2)
height2 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
play_time = frame_num / fps
codec = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('output.mp4', codec, 60, (width2, height2))

frame_count = 0
# 動画終了まで繰り返し
while (cap.isOpened()):
    # フレームを取得
    ret, frame = cap.read()
    if ret == False: break
    frame = cv2.resize(frame, (int(width * 2), int(height * 2)))
    position = frame_count / fps
    position_text = f'{position:.3f}'
    playtime_text = f'{play_time:.3f}'
    cv2.putText(frame,
                text=position_text + '/' + playtime_text,
                org=(20, int(height * 2 - 20)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_4)
    longest_time = 0
    longest_action = ''
    remain_action_text = ''
    for action in action_list:
        start = action[1]
        end = action[2]
        length = end - start
        if start <= position <= end:
            if longest_time < length:
                longest_time = length
                longest_action = action[0]
                remain_action = end - position
                remain_action_text = f'{remain_action:.3f}'

    if len(longest_action) != '' and len(remain_action_text) != 0:
        cv2.putText(frame,
                text=longest_action + ' : ' + remain_action_text,
                org=(40, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(255, 0, 255),
                thickness=2,
                lineType=cv2.LINE_4)

    # フレームを表示
    cv2.imshow("Frame", frame)
    video.write(frame)

    frame_count += 1

    # qキーが押されたら途中終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # 特徴点終わり
    if position >= end_max:
        break


cap.release()
video.release()
cv2.destroyAllWindows()
