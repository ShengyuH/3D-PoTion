""" Utility script for cropping all the frames based on the bounding box

Author: Shengyu Huang
Date: June 2019
"""
from utils_2d import *
from config import configs

PATH_PENN=configs['dir_penn_action']
DIR_LABELS = os.path.join(PATH_PENN,'labels')
DIR_FRAMES = os.path.join(PATH_PENN,'frames')
DIR_CROPPED = os.path.join(PATH_PENN,'cropped')
EDGE=configs['edge']

all_labels = sorted(os.listdir(DIR_LABELS))
count=0
for eachlabel in all_labels:
    if(eachlabel!='.DS_Store'):
        if(count%100==0):
            print(count)
        count+=1
        crn_video = eachlabel.split(".")[0]
        dir_video = os.path.join(DIR_FRAMES, crn_video)
        all_frames = sorted(os.listdir(dir_video))
        annots = sio.loadmat(os.path.join(DIR_LABELS, eachlabel))
        T = annots['nframes'][0][0]
        crn_bbox = annots["bbox"].astype("int")
        dir_cropped = os.path.join(DIR_CROPPED, crn_video)
        if(os.path.exists(dir_cropped)):
            continue
        os.system("mkdir " + dir_cropped)

        # crop images based on bounding box
        for i in range(crn_bbox.shape[0]):
            path_crn_frame = os.path.join(dir_video, all_frames[i])
            img = mpimg.imread(path_crn_frame)
            # bbox
            min_x = max(0, crn_bbox[i, 0] - EDGE)
            min_y = max(0, crn_bbox[i, 1] - EDGE)
            max_x = min(crn_bbox[i, 2] + EDGE, img.shape[1])
            max_y = min(crn_bbox[i, 3] + EDGE, img.shape[0])

            img_cropped = img[min_y:max_y, min_x:max_x, :]
            imageio.imwrite(
                os.path.join(dir_cropped, all_frames[i]), img_cropped
            )