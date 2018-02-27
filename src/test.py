import mv3d
import argparse
import os
import numpy as np
import math
from numpy.linalg import inv
import cv2
from data import preprocess
import time
import pickle

testing_type = 'train' # val, test
img_list_path="/home/saic/MV3D/data/kitti/raw/ImageSets/%s.txt" % (testing_type)

calib_dir = '/home/saic/MV3D/data/kitti/raw/data_object_calib/training/calib'
img_dir = '/home/saic/MV3D/data/kitti/raw/data_object_image_2/training/image_2'
lidar_dir = '/home/saic/MV3D/data/kitti/raw/data_object_velodyne/training/velodyne'
lidar_cache_dir = '/home/saic/MV3D/data/kitti/raw/data_object_velodyne/training/velodyne_cache'

results_img_dir = '/home/saic/MV3D/results/image'
results_dir = '/home/saic/MV3D/results/output'


if not os.path.isdir(lidar_cache_dir):
    os.makedirs(lidar_cache_dir)


#################################################################################


def box3d_to_rgb_box(box3d, Mt):
    # 8 x 4
    Ps = np.hstack(( box3d, np.ones((8,1))) )
    #  4 x 4, 8 x 4 -> 4 x 8
    Projected = np.dot(Mt, Ps.T)
    Projected = Projected[0:2, :] / Projected[2, :]

    # transform to big bound box
    minX = int(np.min(Projected[0, :]))
    maxX = int(np.max(Projected[0, :]))
    minY = int(np.min(Projected[1, :]))
    maxY = int(np.max(Projected[1, :]))

    return (minX, minY, maxX, maxY)


def load_rgb(img_path):
    # may need to do some preprocessing
    img = cv2.imread(img_path)
    # make it into 4d
    img = np.array([img], dtype=np.float32)
    return img


def load_top(lidar_path, lidar_cache_path=None, overwrite_cache=False):
    # set lidar_cache_path to None to disable loading from cache

    if lidar_cache_path == None or os.path.isfile(lidar_cache_path) == False or overwrite_cache == True:
        # load lidar points
        lidar = np.fromfile(lidar_path, np.float32)
        lidar = lidar.reshape((-1, 4))

        # top feature extraction
        top = preprocess.lidar_to_top(lidar)
        # make it into 4d
        top = np.array([top], dtype=np.float32)

        # create cache
        if lidar_cache_path != None:
            with open(lidar_cache_path, 'wb') as f:
                pickle.dump(top, f, -1)
    else:
        # load from cache
        with open(lidar_cache_path, 'rb') as f:
            top = pickle.load(f)

    return top



def load_calib(calib_file_path):
    """
    load P2 (for rgb camera 2), R0_Rect, Tr_velo_to_cam, and compute the velo_to_rgb
    :param frame_tag: e.g.,2011_09_26/2011_09_26_drive_0009_sync/0000000021
    :return: calibration matrix
    """

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
                #print "P2:"
                #print obj.P2

            elif name == 'R0_rect':
                obj.R0_rect = np.zeros((4, 4), dtype = calib_data.dtype)
                obj.R0_rect[:3, :3] = calib_data.reshape((3, 3))
                obj.R0_rect[3, 3] = 1

                #print "R0_rect:"
                #print obj.R0_rect


            elif name == 'Tr_velo_to_cam':
                obj.velo_to_cam = np.hstack((calib_data, [0, 0, 0, 1])).reshape(4, 4)
                #print "velo_to_cam:"
                #print obj.velo_to_cam

        obj.velo_to_rgb = np.dot(obj.P2, np.dot(obj.R0_rect, obj.velo_to_cam))
        obj.cam_to_rgb = np.dot(obj.P2, obj.R0_rect)
        obj.cam_to_velo = inv(obj.velo_to_cam)
    return obj

def box3d_decompose_camcord(box3d):
    # translation
    T_x = np.sum(box3d[0:4, 0], 0) / 4.0
    T_y = np.sum(box3d[0:4, 1], 0) / 4.0
    T_z = np.sum(box3d[0:4, 2], 0) / 4.0

    Point0 = box3d[0, (0, 2)]
    Point1 = box3d[1, (0, 2)]
    Point2 = box3d[2, (0, 2)]

    dis1=np.sum((Point0 - Point1)**2)**0.5
    dis2=np.sum((Point1 - Point2)**2)**0.5

    dis1_is_max=dis1 > dis2

    #length width heigth
    L=np.maximum(dis1, dis2)
    W=np.minimum(dis1, dis2)
    H=np.sum((box3d[0] - box3d[4])**2)**0.5

    # rotation
    yaw=lambda p1,p2: -math.atan2(p2[1]-p1[1],p2[0]-p1[0])

    translation = (T_x, T_y, T_z)
    size = (H, W, L)

    rot_y= yaw(Point0, Point1) if dis1_is_max else yaw(Point1, Point2)

    return translation, size, rot_y

def get_tags(filename):
    with open(filename, 'r') as f:
        tags = f.read().splitlines()
    return tags


def output(boxes3d, probs, file_path, calib_velo_to_cam, calib_velo_to_rgb, img_height, img_width):
    # output box3d to file_path with the format of evaluation file
    # translation n x 3, calib_velo_to_cam 4 x 4
    trash = -1000
    invalid_aos = -10
    N = len(boxes3d)
    with open(file_path, 'w') as f:
        for i in range(N):
            box3d = boxes3d[i]
            # box3d to rgb big bounding box
            minX, minY, maxX, maxY = box3d_to_rgb_box(box3d, calib_velo_to_rgb)
            # clip the cord
            minX = max(0, minX)
            minY = max(0, minY)
            maxX = min(img_width - 1, maxX)
            maxY = min(img_height - 1, maxY)


            # 8 x 3   4 x 4  -> 4 x 8
            box3d = np.dot(calib_velo_to_cam, np.vstack([box3d.T, np.ones([1, 8])]))
            box3d = box3d[:3, :].transpose()


            # decompose the bounding box into H, W, L, translation, and rotation
            translation, size, rot_y = box3d_decompose_camcord(box3d)
            line = "%s %lf %f %d %d %d %d %d %lf %lf %lf %lf %lf %lf %lf %lf\n" % \
                   ('car', trash,  trash, invalid_aos, minX, minY, maxX, maxY, \
                    size[0],size[1],size[2], \
                    translation[0], translation[1], translation[2], \
                     rot_y, probs[i])
            f.write(line)


            """ 
                if (fscanf(fp, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                           str, &trash, &trash, &d.box.alpha, &d.box.x1, &d.box.y1,
                           &d.box.x2, &d.box.y2, &d.h, &d.w, &d.l, &d.t1, &d.t2, &d.t3,
                           &d.ry, &d.thresh)==16) {
            """



if __name__ == '__main__':
    is_debug = True

    parser = argparse.ArgumentParser(description='testing')
    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')
    parser.add_argument('-w', '--weights', type=str, nargs='?', default='',
                        help='set weights tag name')


    args = parser.parse_args()


    print('\n\n{}\n\n'.format(args))

    log_tag = args.tag
    if is_debug:
        # log_tag = 'iter_120000-late_fusion-exclude_neg_gt-focal_loss-bn_fix-regularize'
        # log_tag = 'debug_xcep'
        # log_tag = 'debug_vgg16_normal_loss_samelr'
        # log_tag = 'debug_remove_out_boundary_rgb'
        # log_tag = 'iter_120000'
        log_tag = 'debug'
    if log_tag == 'unknown_tag':
        log_tag = input('Enter log tag : ')
        print('\nSet log tag :"%s" ok !!\n' % log_tag)

    weights_tag = args.weights if args.weights != '' else None


    tags = get_tags(img_list_path)
    test_size = len(tags)


    # init, rgb_shape is not constrained
    snapshot_iters = [None]
    do_image_logging = True
    do_print = True


    for one_snapshot_iter in snapshot_iters:

        predicator = mv3d.Predictor(top_shape=(704, 800, 7), rgb_shape=None, log_tag=log_tag, weights_tag=weights_tag,
                                    iter=one_snapshot_iter)

        one_snapshot_iter_str = str(one_snapshot_iter) if one_snapshot_iter else 'final'
        result_img_dir_prefix = os.path.join(results_img_dir, log_tag, one_snapshot_iter_str, testing_type)
        if not os.path.isdir(result_img_dir_prefix):
            os.makedirs(result_img_dir_prefix)

        result_output_dir_prefix = os.path.join(results_dir, log_tag, one_snapshot_iter_str, testing_type, 'data')
        if not os.path.isdir(result_output_dir_prefix):
            os.makedirs(result_output_dir_prefix)

        for frame_id in tags:
            # load rgb
            img_path = os.path.join(img_dir, "%06d.png" % (int(frame_id)))
            rgb = load_rgb(img_path)
            img_height = rgb.shape[1]
            img_width = rgb.shape[2]

            # load point clouds and feature extraction to top
            lidar_path = os.path.join(lidar_dir, "%06d.bin" % (int(frame_id)))
            # lidar_cache_path = os.path.join(lidar_cache_dir, "%06d.pkl" % (int(frame_id)))
            top = load_top(lidar_path, lidar_cache_path=None)

            # load calib
            calib_path = os.path.join(calib_dir, "%06d.txt" % (int(frame_id)))
            calib = load_calib(calib_path)

            # time the prediction
            # start = time.time()
            # for i in range(200):
            boxes3d, probs = predicator.predict(top, rgb, calib.velo_to_rgb)
            # end = time.time()
            # print("average prediciton time %f" % ((end - start) / 200.0))

            if do_image_logging:
                result_img_path_prefix = os.path.join(result_img_dir_prefix, "%06d" % (int(frame_id)))
                predicator.log_tested_image(result_img_path_prefix, calib.velo_to_rgb, log_rpn=True, frame_tag=frame_id)


            reslut_output_path = os.path.join(result_output_dir_prefix, "%06d.txt" % (int(frame_id)))
            output(boxes3d, probs, reslut_output_path, calib.velo_to_cam, calib.velo_to_rgb, img_height, img_width)

            if do_print:
                # print("frame_id: %d" % (int(frame_id)))
                print("frame_id: %d, detect %d boxes" % (int(frame_id), len(boxes3d)))




