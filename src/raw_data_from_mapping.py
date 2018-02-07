import os
import numpy as np
import cv2
from config import cfg
from numpy.linalg import inv
import sys


class RawData(object):
    def __init__(self, use_raw = False):
        self.mapping_file = cfg.MAPPING_FILE
        self.rand_map = cfg.RAND_MAP
        self.path_prefix = ""
        self.ext = ""
        self.files_path_mapping = {}

        self.use_raw = use_raw
        self.train_val_list = cfg.TRAIN_VAL_LIST


    def get_trainval_mapping(self):
        trainval_list_dict = {}
        with open(self.train_val_list, 'r') as f:
            lines = f.read().splitlines()
            line_list = [[line, os.path.join(self.path_prefix, "%s" % (line)) + self.ext] for line in lines]
            trainval_list_dict = dict(line_list)

        return trainval_list_dict




    def get_paths_mapping(self):
        """
        :return: frame_tag_mapping (key: frame_tag val: file_path)
        """
        reverse_file_dict = {}
        with open(self.rand_map, "r") as f:
            line = f.read().splitlines()[0]
            for i, field in enumerate(line.split(',')):
                reverse_file_dict[int(field)] = os.path.join(self.path_prefix, "%06d" % (i)) + self.ext


        with open(self.mapping_file, "r") as f:
            lines = f.read().splitlines()
            lines_splitted = [line.split() for line in lines]
            frame_tag_lines = [('/'.join(line), index) for index, line in enumerate(lines_splitted)]

        frame_tag_map = {}
        for one_frame_tag_line in frame_tag_lines:
            frame_tag, index = one_frame_tag_line
            frame_tag_map[frame_tag] = reverse_file_dict[int(index) + 1]

        return frame_tag_map

    def get_tags(self):
        tags = [tag for tag in self.files_path_mapping]
        tags.sort()
        return tags


class Image(RawData):

    def __init__(self, use_raw = False):
        RawData.__init__(self, use_raw)
        self.path_prefix = os.path.join(cfg.RAW_DATA_SETS_DIR, "data_object_image_2", "training", "image_2")
        self.ext = ".png"
        if use_raw:
            self.files_path_mapping= self.get_paths_mapping()
        else:
            self.files_path_mapping= self.get_trainval_mapping()



    def load(self, frame_tag):
        return cv2.imread(self.files_path_mapping[frame_tag])



class ObjectAnnotation(RawData):

    def __init__(self, use_raw = False):
        RawData.__init__(self, use_raw)
        self.path_prefix = os.path.join(cfg.RAW_DATA_SETS_DIR, "data_object_label_2", "training", "label_2")
        self.ext = ".txt"

        if use_raw:
            self.files_path_mapping= self.get_paths_mapping()
        else:
            self.files_path_mapping= self.get_trainval_mapping()

    def parseline(self, line):

        obj = type('object_annotation', (), {})
        fields = line.split()

        obj.type, obj.trunc, obj.occlu = fields[0], float(fields[1]), float(fields[2])
        obj.alpha, obj.left, obj.top, obj.right, obj.bottom = float(fields[3]), float(fields[4]), float(fields[5]), \
                                                                float(fields[6]), float(fields[7])
        obj.h, obj.w, obj.l, obj.x, obj.y, obj.z, obj.rot_y = float(fields[8]), float(fields[9]), float(fields[10]), \
                                            float(fields[11]), float(fields[12]), float(fields[13]), float(fields[14])
        return obj


    def load(self, frame_tag):
        """
        load object annotation file, including bounding box, and object label
        :param frame_tag:
        :return:
        """
        annot_path = self.files_path_mapping[frame_tag]
        objs = []
        with open(annot_path, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                if not line:
                    continue
                objs.append(self.parseline(line))

        return objs



class Calibration(RawData):
    def __init__(self, use_raw = False):
        RawData.__init__(self, use_raw)
        self.path_prefix = os.path.join(cfg.RAW_DATA_SETS_DIR, "data_object_calib", "training", "calib")
        self.ext = ".txt"

        if use_raw:
            self.files_path_mapping= self.get_paths_mapping()
        else:
            self.files_path_mapping= self.get_trainval_mapping()

    def load(self, frame_tag):
        """
        load P2 (for rgb camera 2), R0_Rect, Tr_velo_to_cam, and compute the velo_to_rgb
        :param frame_tag: e.g.,2011_09_26/2011_09_26_drive_0009_sync/0000000021
        :return: calibration matrix
        """

        calib_file_path = self.files_path_mapping[frame_tag]
        obj = type('calib', (), {})
        with open(calib_file_path, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                if not line:
                    continue
                fields = line.split(':')
                name, calib_str = fields[0], fields[1]
                calib_data = np.fromstring(calib_str, sep=' ', dtype=np.float32)

                if name == 'P2':
                    obj.P2 = np.hstack((calib_data, [0, 0, 0, 1])).reshape(4, 4)

                elif name == 'R0_rect':
                    obj.R0_rect = np.zeros((4, 4), dtype = calib_data.dtype)
                    obj.R0_rect[:3, :3] = calib_data.reshape((3, 3))
                    obj.R0_rect[3, 3] = 1

                elif name == 'Tr_velo_to_cam':
                    obj.velo_to_cam = np.hstack((calib_data, [0, 0, 0, 1])).reshape(4, 4)

            obj.velo_to_rgb = np.dot(obj.P2, np.dot(obj.R0_rect, obj.velo_to_cam))
            obj.cam_to_rgb = np.dot(obj.P2, obj.R0_rect)
            obj.cam_to_velo = inv(obj.velo_to_cam)
        return obj



class Lidar(RawData):
    def __init__(self, use_raw = False):
        RawData.__init__(self, use_raw)
        self.path_prefix = os.path.join(cfg.RAW_DATA_SETS_DIR, "data_object_velodyne", "training", "velodyne")
        self.ext = ".bin"

        if use_raw:
            self.files_path_mapping= self.get_paths_mapping()
        else:
            self.files_path_mapping= self.get_trainval_mapping()

    def load(self, frame_tag):
        lidar =np.fromfile(self.files_path_mapping[frame_tag], np.float32)
        return lidar.reshape((-1, 4))



if __name__ == '__main__':
    pass
