import net.processing.boxes3d  as box
from multiprocessing import Process, Queue as Queue, Value, Array
import time

import config
import cv2
import net.utility.draw as draw
from raw_data_from_mapping import *
from train_val_splitter_kitti_benchmark import TrainingValDataSplitter
import pickle
import array
from sklearn.utils import shuffle
import threading
import data


def draw_bbox_on_rgb(rgb, boxes3d, one_frame_tag, calib_velo_to_rgb):
    img = draw.draw_box3d_on_camera(rgb, boxes3d, calib_velo_to_rgb)
    #new_size = (img.shape[1] // 3, img.shape[0] // 3)
    #img = cv2.resize(img, new_size)
    path = os.path.join(config.cfg.LOG_DIR, 'test', 'rgb', '%s.png' % one_frame_tag.replace('/', '_'))
    cv2.imwrite(path, img)
    print('write %s finished' % path)


def draw_bbox_on_lidar_top(top, boxes3d, one_frame_tag):
    path = os.path.join(config.cfg.LOG_DIR, 'test', 'top', '%s.png' % one_frame_tag.replace('/', '_'))
    top_image = data.draw_top_image(top)
    top_image = box.draw_box3d_on_top(top_image, boxes3d, color=(0, 0, 80))
    cv2.imwrite(path, top_image)
    print('write %s finished' % path)


use_thread = True


class BatchLoading_kitti_benchmark:
    def __init__(self, tags, queue_size=20, require_shuffle=False,
                 require_log=False, is_testset=False, random_num=666, is_flip=False):

        self.is_testset = is_testset
        self.shuffled = require_shuffle
        self.random_num = random_num
        self.preprocess = data.Preprocess()

        self.raw_img = Image()
        self.raw_lidar = Lidar()
        self.raw_calib = Calibration()
        self.raw_object_annot = ObjectAnnotation()


        self.is_flip = is_flip

        self.tags = tags
        if self.shuffled:
            self.tags = shuffle(tags, random_state=self.random_num)

        self.tag_index = 0
        self.size = len(self.tags)

        self.require_log = require_log
        self.flip_axis = 1 # if axis=1, flip from y=0. If axis=0, flip from x=0
        self.flip_rate = 2 # if flip_rate is 2, means every two frames

        self.cache_size = queue_size
        self.loader_need_exit = Value('i', 0)

        if use_thread:
            self.prepr_data = []
            self.lodaer_processing = threading.Thread(target=self.loader)
        else:
            self.preproc_data_queue = Queue()
            self.buffer_blocks = [Array('h', 41246691) for i in range(queue_size)]
            self.blocks_usage = Array('i', range(queue_size))
            self.lodaer_processing = Process(target=self.loader)
        self.lodaer_processing.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.loader_need_exit.value = True
        if self.require_log: print('set loader_need_exit True')
        self.lodaer_processing.join()
        if self.require_log: print('exit lodaer_processing')

    def keep_gt_inside_range(self, train_gt_labels, train_gt_boxes3d):
        train_gt_labels = np.array(train_gt_labels, dtype=np.int32)
        train_gt_boxes3d = np.array(train_gt_boxes3d, dtype=np.float32)
        if train_gt_labels.shape[0] == 0:
            return False, None, None
        assert train_gt_labels.shape[0] == train_gt_boxes3d.shape[0]

        # get limited train_gt_boxes3d and train_gt_labels.
        keep = np.zeros((len(train_gt_labels)), dtype=bool)

        for i in range(len(train_gt_labels)):
            if box.box3d_in_top_view(train_gt_boxes3d[i]):
                keep[i] = 1

        # if all targets are out of range in selected top view, return True.
        if np.sum(keep) == 0:
            return False, None, None

        train_gt_labels = train_gt_labels[keep]
        train_gt_boxes3d = train_gt_boxes3d[keep]

        return True, train_gt_labels, train_gt_boxes3d

    def load_from_one_tag(self, one_frame_tag):
        rgb = self.raw_img.load(one_frame_tag)
        lidar = self.raw_lidar.load(one_frame_tag)

        object_annots = self.raw_object_annot.load(one_frame_tag) if not self.is_testset else None
        calib = self.raw_calib.load(one_frame_tag)

        return  rgb, lidar, object_annots, calib

    def preprocess_one_frame(self, rgb, lidar, calib, object_annots):
        rgb = self.preprocess.rgb(rgb)
        top = self.preprocess.lidar_to_top(lidar)

        if self.is_testset:
            return rgb, top, None, None


        boxes3d = [self.preprocess.bbox3d_cam_to_velo(obj, calib.cam_to_velo) for obj in object_annots]
        labels = [self.preprocess.label(obj) for obj in object_annots]
        # flip in y axis.
        if self.is_flip and len(boxes3d) > 0:
            if self.tag_index % self.flip_rate == 1:
                top, rgb, boxes3d = self.preprocess.flip(rgb, top, boxes3d, axis=1)
            elif self.tag_index % self.flip_rate == 2:
                top, rgb, boxes3d = self.preprocess.flip(rgb, top, boxes3d, axis=0)
        return rgb, top, boxes3d, labels

    def get_shape(self):
        # todo for tracking, it means wasted a frame which will cause offset.
        train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d, _, _ = self.load()
        top_shape = train_tops[0].shape
        front_shape = train_fronts[0].shape
        rgb_shape = train_rgbs[0].shape

        return top_shape, front_shape, rgb_shape

    def data_preprocessed(self):
        # only feed in frames with ground truth labels and bboxes during training, or the training nets will break.
        skip_frames = True
        while skip_frames:
            fronts = []
            frame_tag = self.tags[self.tag_index]
            rgb, lidar, object_annots, calib = self.load_from_one_tag(frame_tag)
            rgb, top, boxes3d, labels = self.preprocess_one_frame(rgb, lidar, calib, object_annots)
            if self.require_log and not self.is_testset:
                draw_bbox_on_rgb(rgb, boxes3d, frame_tag, calib.velo_to_rgb)
                draw_bbox_on_lidar_top(top, boxes3d, frame_tag)

            self.tag_index += 1

            # reset self tag_index to 0 and shuffle tag list
            if self.tag_index >= self.size:
                self.tag_index = 0
                if self.shuffled:
                    self.tags = shuffle(self.tags, random_state=self.random_num)
            skip_frames = False

            # only feed in frames with ground truth labels and bboxes during training, or the training nets will break.
            if not self.is_testset:
                is_gt_inside_range, batch_gt_labels_in_range, batch_gt_boxes3d_in_range = \
                    self.keep_gt_inside_range(labels, boxes3d)
                labels = batch_gt_labels_in_range
                boxes3d = batch_gt_boxes3d_in_range
                # if no gt labels inside defined range, discard this training frame.
                # now skip the training samples without positive gt
                if not is_gt_inside_range or np.sum(labels) == 0:
                    skip_frames = True

        return np.array([rgb]), np.array([top]), np.array([fronts]), np.array([labels]), \
               np.array([boxes3d]), frame_tag, calib

    def find_empty_block(self):
        idx = -1
        for i in range(self.cache_size):
            if self.blocks_usage[i] == 1:
                continue
            else:
                idx = i
                break
        return idx

    def loader(self):
        if use_thread:
            while self.loader_need_exit.value == 0:

                if len(self.prepr_data) >= self.cache_size:
                    time.sleep(1)
                    # print('sleep ')
                else:
                    self.prepr_data = [(self.data_preprocessed())] + self.prepr_data
        else:
            while self.loader_need_exit.value == 0:
                empty_idx = self.find_empty_block()
                if empty_idx == -1:
                    time.sleep(1)
                    # print('sleep ')
                else:
                    prepr_data = (self.data_preprocessed())
                    dumps = pickle.dumps(prepr_data)
                    length = len(dumps)
                    self.buffer_blocks[empty_idx][0:length] = dumps[0:length]

                    self.preproc_data_queue.put({
                        'index': empty_idx,
                        'length': length
                    })

        if self.require_log: print('loader exit')

    def load(self):
        if use_thread:
            while len(self.prepr_data) == 0:
                time.sleep(1)
            data_ori = self.prepr_data.pop()


        else:

            # print('self.preproc_data_queue.qsize() = ', self.preproc_data_queue.qsize())
            info = self.preproc_data_queue.get(block=True)
            length = info['length']
            block_index = info['index']
            dumps = self.buffer_blocks[block_index][0:length]

            # set flag
            self.blocks_usage[block_index] = 0

            # convert to bytes string
            dumps = array.array('B', dumps).tostring()
            data_ori = pickle.loads(dumps)

        return data_ori

    def get_frame_info(self):
        return self.tags[self.tag_index]


if __name__ == '__main__':


    train_n_val_dataset = [
        '2011_09_26/2011_09_26_drive_0001_sync',
        '2011_09_26/2011_09_26_drive_0002_sync',
        '2011_09_26/2011_09_26_drive_0005_sync',
        '2011_09_26/2011_09_26_drive_0009_sync',
        '2011_09_26/2011_09_26_drive_0011_sync',
        '2011_09_26/2011_09_26_drive_0013_sync',
        '2011_09_26/2011_09_26_drive_0014_sync',
        '2011_09_26/2011_09_26_drive_0015_sync',
        '2011_09_26/2011_09_26_drive_0017_sync',
        '2011_09_26/2011_09_26_drive_0018_sync',
        '2011_09_26/2011_09_26_drive_0019_sync',
        '2011_09_26/2011_09_26_drive_0020_sync',
        '2011_09_26/2011_09_26_drive_0022_sync',
        '2011_09_26/2011_09_26_drive_0023_sync',
        '2011_09_26/2011_09_26_drive_0027_sync',
        '2011_09_26/2011_09_26_drive_0028_sync',
        '2011_09_26/2011_09_26_drive_0029_sync',
        '2011_09_26/2011_09_26_drive_0032_sync',
        '2011_09_26/2011_09_26_drive_0035_sync',
        '2011_09_26/2011_09_26_drive_0036_sync',
        '2011_09_26/2011_09_26_drive_0039_sync',
        '2011_09_26/2011_09_26_drive_0046_sync',
        '2011_09_26/2011_09_26_drive_0048_sync',
        '2011_09_26/2011_09_26_drive_0051_sync',
        '2011_09_26/2011_09_26_drive_0052_sync',
        '2011_09_26/2011_09_26_drive_0056_sync',
        '2011_09_26/2011_09_26_drive_0057_sync',
        '2011_09_26/2011_09_26_drive_0059_sync',
        '2011_09_26/2011_09_26_drive_0060_sync',
        '2011_09_26/2011_09_26_drive_0061_sync',
        '2011_09_26/2011_09_26_drive_0064_sync',
        '2011_09_26/2011_09_26_drive_0070_sync',
        '2011_09_26/2011_09_26_drive_0079_sync',
        '2011_09_26/2011_09_26_drive_0084_sync',
        '2011_09_26/2011_09_26_drive_0086_sync',
        '2011_09_26/2011_09_26_drive_0087_sync',
        '2011_09_26/2011_09_26_drive_0091_sync',
        '2011_09_26/2011_09_26_drive_0093_sync',
        '2011_09_26/2011_09_26_drive_0095_sync',
        '2011_09_26/2011_09_26_drive_0096_sync',
        '2011_09_26/2011_09_26_drive_0101_sync',
        '2011_09_26/2011_09_26_drive_0104_sync',
        '2011_09_26/2011_09_26_drive_0106_sync',
        '2011_09_26/2011_09_26_drive_0113_sync',
        '2011_09_26/2011_09_26_drive_0117_sync',
        '2011_09_28/2011_09_28_drive_0001_sync',
        '2011_09_28/2011_09_28_drive_0002_sync',
        '2011_09_28/2011_09_28_drive_0016_sync',
        '2011_09_28/2011_09_28_drive_0021_sync',
        '2011_09_28/2011_09_28_drive_0034_sync',
        '2011_09_28/2011_09_28_drive_0035_sync',
        '2011_09_28/2011_09_28_drive_0037_sync',
        '2011_09_28/2011_09_28_drive_0038_sync',
        '2011_09_28/2011_09_28_drive_0039_sync',
        '2011_09_28/2011_09_28_drive_0043_sync',
        '2011_09_28/2011_09_28_drive_0045_sync',
        '2011_09_28/2011_09_28_drive_0047_sync',
        '2011_09_28/2011_09_28_drive_0053_sync',
        '2011_09_28/2011_09_28_drive_0054_sync',
        '2011_09_28/2011_09_28_drive_0057_sync',
        '2011_09_28/2011_09_28_drive_0065_sync',
        '2011_09_28/2011_09_28_drive_0066_sync',
        '2011_09_28/2011_09_28_drive_0068_sync',
        '2011_09_28/2011_09_28_drive_0070_sync',
        '2011_09_28/2011_09_28_drive_0071_sync',
        '2011_09_28/2011_09_28_drive_0075_sync',
        '2011_09_28/2011_09_28_drive_0077_sync',
        '2011_09_28/2011_09_28_drive_0078_sync',
        '2011_09_28/2011_09_28_drive_0080_sync',
        '2011_09_28/2011_09_28_drive_0082_sync',
        '2011_09_28/2011_09_28_drive_0086_sync',
        '2011_09_28/2011_09_28_drive_0087_sync',
        '2011_09_28/2011_09_28_drive_0089_sync',
        '2011_09_28/2011_09_28_drive_0090_sync',
        '2011_09_28/2011_09_28_drive_0094_sync',
        '2011_09_28/2011_09_28_drive_0095_sync',
        '2011_09_28/2011_09_28_drive_0096_sync',
        '2011_09_28/2011_09_28_drive_0098_sync',
        '2011_09_28/2011_09_28_drive_0100_sync',
        '2011_09_28/2011_09_28_drive_0102_sync',
        '2011_09_28/2011_09_28_drive_0103_sync',
        '2011_09_28/2011_09_28_drive_0104_sync',
        '2011_09_28/2011_09_28_drive_0106_sync',
        '2011_09_28/2011_09_28_drive_0108_sync',
        '2011_09_28/2011_09_28_drive_0110_sync',
        '2011_09_28/2011_09_28_drive_0113_sync',
        '2011_09_28/2011_09_28_drive_0117_sync',
        '2011_09_28/2011_09_28_drive_0119_sync',
        '2011_09_28/2011_09_28_drive_0121_sync',
        '2011_09_28/2011_09_28_drive_0122_sync',
        '2011_09_28/2011_09_28_drive_0125_sync',
        '2011_09_28/2011_09_28_drive_0126_sync',
        '2011_09_28/2011_09_28_drive_0128_sync',
        '2011_09_28/2011_09_28_drive_0132_sync',
        '2011_09_28/2011_09_28_drive_0134_sync',
        '2011_09_28/2011_09_28_drive_0135_sync',
        '2011_09_28/2011_09_28_drive_0136_sync',
        '2011_09_28/2011_09_28_drive_0138_sync',
        '2011_09_28/2011_09_28_drive_0141_sync',
        '2011_09_28/2011_09_28_drive_0143_sync',
        '2011_09_28/2011_09_28_drive_0145_sync',
        '2011_09_28/2011_09_28_drive_0146_sync',
        '2011_09_28/2011_09_28_drive_0149_sync',
        '2011_09_28/2011_09_28_drive_0153_sync',
        '2011_09_28/2011_09_28_drive_0154_sync',
        '2011_09_28/2011_09_28_drive_0155_sync',
        '2011_09_28/2011_09_28_drive_0156_sync',
        '2011_09_28/2011_09_28_drive_0160_sync',
        '2011_09_28/2011_09_28_drive_0161_sync',
        '2011_09_28/2011_09_28_drive_0162_sync',
        '2011_09_28/2011_09_28_drive_0165_sync',
        '2011_09_28/2011_09_28_drive_0166_sync',
        '2011_09_28/2011_09_28_drive_0167_sync',
        '2011_09_28/2011_09_28_drive_0168_sync',
        '2011_09_28/2011_09_28_drive_0171_sync',
        '2011_09_28/2011_09_28_drive_0174_sync',
        '2011_09_28/2011_09_28_drive_0177_sync',
        '2011_09_28/2011_09_28_drive_0179_sync',
        '2011_09_28/2011_09_28_drive_0183_sync',
        '2011_09_28/2011_09_28_drive_0184_sync',
        '2011_09_28/2011_09_28_drive_0185_sync',
        '2011_09_28/2011_09_28_drive_0186_sync',
        '2011_09_28/2011_09_28_drive_0187_sync',
        '2011_09_28/2011_09_28_drive_0191_sync',
        '2011_09_28/2011_09_28_drive_0192_sync',
        '2011_09_28/2011_09_28_drive_0195_sync',
        '2011_09_28/2011_09_28_drive_0198_sync',
        '2011_09_28/2011_09_28_drive_0199_sync',
        '2011_09_28/2011_09_28_drive_0201_sync',
        '2011_09_28/2011_09_28_drive_0204_sync',
        '2011_09_28/2011_09_28_drive_0205_sync',
        '2011_09_28/2011_09_28_drive_0208_sync',
        '2011_09_28/2011_09_28_drive_0209_sync',
        '2011_09_28/2011_09_28_drive_0214_sync',
        '2011_09_28/2011_09_28_drive_0216_sync',
        '2011_09_28/2011_09_28_drive_0220_sync',
        '2011_09_28/2011_09_28_drive_0222_sync',
        '2011_09_29/2011_09_29_drive_0004_sync',
        '2011_09_29/2011_09_29_drive_0026_sync',
        '2011_09_29/2011_09_29_drive_0071_sync',
        '2011_10_03/2011_10_03_drive_0047_sync',
    ]

    # shuffle bag list or same kind of bags will only be in training or validation set.
    train_n_val_dataset = shuffle(train_n_val_dataset, random_state=666)
    data_splitter = TrainingValDataSplitter(train_n_val_dataset)

    with BatchLoading_kitti_benchmark(tags=data_splitter.training_tags, require_log=False, \
                                    require_shuffle=True, random_num=np.random.randint(100)) as bl:
        time.sleep(5)
        for i in range(2):
            t0 = time.time()
            loaded_data = bl.load()
            print('use time =', time.time() - t0)
            batch_rgb_images, batch_top_view, batch_front_view, \
                batch_gt_labels, batch_gt_boxes3d, frame_id, calib = loaded_data
            #print("batch_gt_labels:")
            #print(batch_gt_labels)
            #print("batch_gt_boxes3d:")
            #print(batch_gt_boxes3d)
            #time.sleep(3)
            # modify projection matrix
            #config.cfg.MATIRIX_velo_to_rgb = calib.velo_to_rgb
            #print(config.MATRIX_velo_to_rgb)

            #print(frame_id)
            #print('---------------------')

        print('Done')
