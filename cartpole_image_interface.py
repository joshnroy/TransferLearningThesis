import numpy as np
import cv2
import os

def get_first_img(pics_folder="randpics/"):
    return cv2.imread(pics_folder + "observation_0_0.jpg", flags=cv2.IMREAD_COLOR)

def read_file(i_episode, j_step, pics_folder="randpics/"):
    filename = pics_folder + "observation_" + str(i_episode) + "_" + \
    str(j_step) + ".jpg"
    if os.path.isfile(filename):
        img = cv2.imread(filename)
        img = img.flatten()
        return img
    else:
        return None

def read_next_file():
    i_episode = 0
    j_step = 0
    ret = read_file(i_episode, j_step)
    while True:
        while ret is not None:
            while ret is not None:
                yield ret
                j_step += 1
                ret = read_file(i_episode, j_step)
            i_episode += 1
            j_step = 0
            ret = read_file(i_episode, j_step)
        i_episode = 0
        j_step = 0
        ret = read_file(i_episode, j_step)


def read_next_batch(batch_size):
    imggen = read_next_file()
    img = next(imggen)
    while True:
        batch = []
        for _ in range(batch_size):
            if img is None:
                return None
            else:
                batch.append(img)
                img = next(imggen)
        yield np.array(batch)