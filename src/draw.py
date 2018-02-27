# load results file and draw on rgb
import os
import config as cfg
import numpy as np
import cv2
from numpy.linalg import inv



results_dir = '/home/saic/MV3D/results/data'
results_img_dir = '/home/saic/MV3D/results_img'

img_list_path='/home/saic/MV3D/data/kitti/raw/ImageSets/val.txt'
calib_dir = '/home/saic/MV3D/data/kitti/raw/data_object_calib/training/calib'
img_dir = '/home/saic/MV3D/data/kitti/raw/data_object_image_2/training/image_2'



def get_tags(filename):
    with open(filename, 'r') as f:
        tags = f.read().splitlines()
    return tags

def parseline(line):

    obj = type('object_annotation', (), {})
    fields = line.split()

    obj.type, obj.trunc, obj.occlu = fields[0], float(fields[1]), float(fields[2])
    obj.alpha, obj.left, obj.top, obj.right, obj.bottom = float(fields[3]), float(fields[4]), float(fields[5]), \
                                                            float(fields[6]), float(fields[7])
    obj.h, obj.w, obj.l, obj.x, obj.y, obj.z, obj.rot_y = float(fields[8]), float(fields[9]), float(fields[10]), \
                                        float(fields[11]), float(fields[12]), float(fields[13]), float(fields[14])
    return obj

def load_img(path):
    return cv2.imread(path)


def load_annot(annot_path):
    """
    load object annotation file, including bounding box, and object label
    :param frame_tag:
    :return:
    """
    objs = []
    with open(annot_path, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            if not line:
                continue
            objs.append(parseline(line))

    return objs

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


def box3d_compose_in_camera_cord_kitti(x, y, z, w, h, l, rot_y):
    """
    only support compose one box, return 3 x 8
    """
    reference_3dbox = np.array([  # in camera coordinates around zero point and without orientation yet\
        [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
        [0.0, 0.0, 0.0, 0.0, -h, -h, -h, -h], \
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]])
    #print "reference:"
    #print reference_3dbox

    # re-create 3D bounding box in camera coordinate system
    rotMat = np.array([ \
        [np.cos(rot_y), 0.0, np.sin(rot_y)], \
        [0.0, 1.0, 0.0], \
        [-np.sin(rot_y), 0.0, np.cos(rot_y)]])

    cornerPosInCamera = np.dot(rotMat, reference_3dbox) + np.tile(np.array([x, y, z]), (8, 1)).T
    #print "after rotation"
    #print cornerPosInCamera

    return cornerPosInCamera


def bbox3d_cam_to_velo(obj, cam_to_velo):
    # cam_corners: 3 x 8   cam_to_velo: 4 x 4
    cam_corners = box3d_compose_in_camera_cord_kitti(obj.x, obj.y, obj.z, obj.w, obj.h, obj.l, obj.rot_y)
    cam_corners = np.vstack((cam_corners, np.ones((1, 8), dtype=cam_corners.dtype)))
    # 4 x 8
    box3d = np.dot(cam_to_velo, cam_corners)
    box3d = box3d[:3, :]
    # 3 x 8
    return box3d.T

def box3d_to_rgb_box(boxes3d, Mt):
    num  = len(boxes3d)
    projections = np.zeros((num,8,2),  dtype=np.int32)
    for n in range(num):
        box3d = boxes3d[n]
        # 8 x 4
        Ps = np.hstack(( box3d, np.ones((8,1))) )
        #  4 x 4, 8 x 4 -> 4 x 8
        Projected = np.dot(Mt, Ps.T)
        Projected = Projected[0:2, :] / Projected[2, :]
        projections[n] = Projected.T


    return projections

def box3d_camcord_to_rgb_box(boxes3d, Mt):
    num  = len(boxes3d)
    projections = np.zeros((num,8,2),  dtype=np.int32)
    for n in range(num):
        box3d = boxes3d[n].T
        # 8 x 4
        Ps = np.hstack(( box3d, np.ones((8,1))) )
        #  4 x 4, 8 x 4 -> 4 x 8
        Projected = np.dot(Mt, Ps.T)
        Projected = Projected[0:2, :] / Projected[2, :]
        projections[n] = Projected.T


    return projections


def draw_rgb_projections(image, projections, color=(255,0,255), thickness=2, darker=1.0):

    img = (image.copy()*darker).astype(np.uint8)
    num=len(projections)
    for n in range(num):
        qs = projections[n]
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

            i,j=k+4,(k+1)%4 + 4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

            i,j=k,k+4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

    return img

def draw_box3d_on_camera(rgb, boxes3d, calib_velo_to_rgb, color=(255, 0, 255), thickness=1, text_lables=[]):
    projections = box3d_to_rgb_box(boxes3d, calib_velo_to_rgb)
    rgb = draw_rgb_projections(rgb, projections, color=color, thickness=thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i,text in enumerate(text_lables):
        text_pos = (np.min(projections[i,:, 0]), max(np.min(projections[i,:, 1]), 15) )
        cv2.putText(rgb, text, text_pos, font, 0.7, (0, 255, 100), 1, cv2.LINE_AA)

    return rgb

def draw_bbox_on_rgb(rgb, boxes3d, calib_velo_to_rgb, path):
    img = draw_box3d_on_camera(rgb, boxes3d, calib_velo_to_rgb)
    cv2.imwrite(path, img)

def draw_2d_bboxes_on_rgb(rgb, boxes, path, color=(255, 0, 0), thickness=2):
    for box in boxes:
        rgb = cv2.rectangle(rgb, (box[0], box[1]), (box[2], box[3]), color, thickness)
    cv2.imwrite(path, rgb)

def draw_bbox_camcord_on_rgb(rgb, boxes3d, calib_cam_to_rgb, path, color=(255, 0, 255), thickness=1, text_lables=[]):
    projections = box3d_camcord_to_rgb_box(boxes3d, calib_cam_to_rgb)
    rgb = draw_rgb_projections(rgb, projections, color=color, thickness=thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i,text in enumerate(text_lables):
        text_pos = (np.min(projections[i,:, 0]), max(np.min(projections[i,:, 1]), 15) )
        cv2.putText(rgb, text, text_pos, font, 0.7, (0, 255, 100), 1, cv2.LINE_AA)

    cv2.imwrite(path, rgb)

if __name__ == '__main__':
    tags = get_tags(img_list_path)
    test_size = len(tags)
    for frame_id in tags:
        print(frame_id)

        result_path = os.path.join(results_dir, "%06d.txt" % (int(frame_id)))
        result_img_path = os.path.join(results_img_dir, "%06d.png" % (int(frame_id)))

        # read in the results
        boxes3d = []
        boxes2d = []
        with open(result_path, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                fds = line.split()
                # %s %lf %lf %lf %d(minX) %d(minY) %d(maxX) %d(maxY) %lf(h) %lf(w) %lf(l) %lf(x) %lf(y) %lf(z) %lf(rot_y) %lf(prob)
                minX, minY, maxX, maxY = int(fds[4]), int(fds[5]), int(fds[6]), int(fds[7])
                obj_type, h, w, l, x, y, z, rot, prob = fds[0], \
                                                        float(fds[8]), float(fds[9]), float(fds[10]), \
                                                        float(fds[11]), float(fds[12]), float(fds[13]), \
                                                        float(fds[14]), float(fds[15])

                boxes3d.append(box3d_compose_in_camera_cord_kitti(x, y, z, w, h, l, rot))
                boxes2d.append((minX, minY, maxX, maxY))

            # load image
            img_path = os.path.join(img_dir, "%06d.png" % (int(frame_id)))
            img = load_img(img_path)

            # load calibration matrix
            calib_path = os.path.join(calib_dir, "%06d.txt" % (int(frame_id)))
            calib = load_calib(calib_path)

            # draw 3d bboxes on rgb
            draw_bbox_camcord_on_rgb(img, boxes3d, calib.cam_to_rgb, result_img_path)

            # draw  2d bboxes on rgb
            result_2d_img_path = os.path.join(results_img_dir, "%06d_2d.png" % (int(frame_id)))
            draw_2d_bboxes_on_rgb(img, boxes2d, result_2d_img_path)


