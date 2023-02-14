import os
import json
import glob
import cv2


def main():
    label_json = open('./annotations/label_define.json', 'r')
    label_json_dict = json.load(label_json)
    label_json.close()

    thumos_json = open('../thumos/annotations/thumos14.json', 'r')
    thumos_json_dict = json.load(thumos_json)
    thumos_json.close()
    # print(thumos_json_dict)

    fake_dict = dict({'version': 'Thumos14-30fps'})
    file_database = dict()
    json_files = glob.glob('./annotations/*.json')
    json_files.sort()
    file_num = len(json_files)
    training_num = int(file_num * 0.7)
    i = 0
    for json_file in json_files:
        if not json_file == './annotations/label_define.json':
            movie_file = './movies/' + os.path.splitext(os.path.basename(json_file))[0] + '.mp4'
            cap = cv2.VideoCapture(movie_file)
            fps = (cap.get(cv2.CAP_PROP_FPS))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            movie_duration = frames / fps
            cap.release()
            file = open(json_file, 'r')
            json_file_dict = json.load(file)
            file.close()

            annotations_list = []
            for area in json_file_dict:
                if area != 'database':
                    area_annotation = json_file_dict[area]['annotations']
                    label = area_annotation['label']
                    label_dict = label_json_dict['label']
                    label_num = int([k for k, v in label_dict.items() if v == label][0])
                    area_segments = area_annotation['segments']
                    min_time = 10000
                    max_time = 0
                    for area_segment in area_segments:
                        time = area_segment['time']
                        if time < min_time:
                            min_time = time
                        if time > max_time:
                            max_time = time
                max_frame = int(max_time * fps)
                min_frame = int(min_time * fps)
                duration_time = [min_time, max_time]
                duration_frame = [min_frame, max_frame]
                annotation_dist = dict({'label': label, 'segment': duration_time, 'segment(frames)': duration_frame,
                                        'label_id': label_num})
                annotations_list.append(annotation_dist)
            if i >= training_num:
                subset_str = 'Validation'
            else:
                subset_str = 'Test'

            movie_file_dict = dict(
                {'subset': subset_str, 'duration': movie_duration, 'fps': fps, 'annotations': annotations_list})
            file_database_name = os.path.splitext(os.path.basename(json_file))[0]
            file_database.update(dict({file_database_name: movie_file_dict}))
            i += 1

    fake_dict['database'] = file_database
    file = open('next_all.json', 'w')
    json.dump(fake_dict, file, indent=4, ensure_ascii=False)
    file.close()


if __name__ == '__main__':
    main()
