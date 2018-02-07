import mv3d
from config import cfg
from batch_loading_kitti_benchmark import BatchLoading_kitti_benchmark as BatchLoading
import argparse
import os
import numpy as np
import math

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
    yaw=lambda p1,p2: math.atan2(p2[1]-p1[1],p2[0]-p1[0])

    translation = np.array([T_x,T_y,T_z])
    size = np.array([H,W,L])
    rot_y = yaw(Point0,Point1) if dis1_is_max else yaw(Point1,Point2)

    return translation, size, rot_y

def get_test_tags():
    test_tags = []
    with open(cfg.TEST_LIST, 'r') as f:
        test_tags = f.read().splitlines()
    return test_tags


def output(boxes3d, probs, file_path, calib_velo_to_cam):
    # output box3d to file_path with the format of evaluation file
    # translation n x 3, calib_velo_to_cam 4 x 4
    trash = -1000
    N = len(boxes3d)
    with open(file_path, 'w') as f:
        for i in xrange(N):
            box3d = boxes3d[i]
            # 8 x 3   4 x 4
            box3d = np.dot(calib_velo_to_cam, np.vstack(box3d.T, np.ones(1, 8)))
            box3d = box3d[:3, :].transpose()

            # decompose the bounding box into H, W, L, translation, and rotation
            translation, size, rot_y = box3d_decompose_camcord(box3d)
            line = "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n" % \
                   ('Car', trash,  trash, trash, trash, trash, trash, trash, \
                    size[0],size[1],size[1], \
                    translation[i][0], translation[i][1], translation[i][2], \
                     rot_y, probs[i])
            f.write(line)


            """ 
                if (fscanf(fp, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                           str, &trash, &trash, &d.box.alpha, &d.box.x1, &d.box.y1,
                           &d.box.x2, &d.box.y2, &d.h, &d.w, &d.l, &d.t1, &d.t2, &d.t3,
                           &d.ry, &d.thresh)==16) {
            """





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='testing')
    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')
    parser.add_argument('-w', '--weights', type=str, nargs='?', default='',
                        help='set weights tag name')

    parser.add_argument('-d', '--dir', type=str, nargs='?', default='',
                        help='set results dir')

    args = parser.parse_args()


    print('\n\n{}\n\n'.format(args))

    log_tag = args.tag
    if log_tag == 'unknown_tag':
        log_tag = input('Enter log tag : ')
        print('\nSet log tag :"%s" ok !!\n' % log_tag)

    weights_tag = args.weights if args.weights != '' else None

    results_dir = args.dir
    if results_dir == '':
        results_dir = input('Enter results dir : ')
        print('\nSet results dir :"%s" ok !!\n' % results_dir)

    ext = ".txt"

    test_tags = get_test_tags()
    test_size = len(test_tags)
    with BatchLoading(test_tags, require_shuffle=False, is_testset=True) as dataset_loader:
        # first load to get shape and predict
        rgb, top, front, _, _, frame_id, calib = \
            dataset_loader.load()
        top_shape = rgb[0].shape
        front_shape = front[0].shape
        rgb_shape = top[0].shape

        predicator = mv3d.Predictor(top_shape, front_shape, rgb_shape, log_tag = log_tag, weights_tag = weights_tag)
        boxes3d, probs = predicator.predict(top, front, rgb, calib.velo_to_rgb)
        output_file = os.path.join(results_dir, frame_id + ext)
        output(boxes3d, probs, output_file, calib.velo_to_cam)


        # predict for the remaining
        for i in xrange(1, test_size):
            rgb, top, front, _, _, frame_id, calib_velo_to_rgb = \
                dataset_loader.load()
            boxes3d, prob = predicator.predict(top, front, rgb, calib_velo_to_rgb)
            output_file = os.path.join(results_dir, frame_id + ext)
            output(boxes3d, probs, output_file, calib.velo_to_cam)



