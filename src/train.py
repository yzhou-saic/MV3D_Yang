import numpy as np
import mv3d
import mv3d_net
import glob
from sklearn.utils import shuffle
import argparse
import os
from batch_loading_kitti_benchmark import BatchLoading_kitti_benchmark as BatchLoading
from train_val_splitter_kitti_benchmark import TrainingValDataSplitter
import config


def str2bool(v):
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')

    all = '%s,%s,%s' % (mv3d_net.top_view_rpn_name, mv3d_net.imfeature_net_name, mv3d_net.fusion_net_name)

    parser.add_argument('-w', '--weights', type=str, nargs='?', default='',
                        help='use pre trained weights example: -w "%s" ' % (all))

    parser.add_argument('-t', '--targets', type=str, nargs='?', default=all,
                        help='train targets example: -w "%s" ' % (all))

    parser.add_argument('-i', '--max_iter', type=int, nargs='?', default=1000,
                        help='max count of train iter')

    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')

    parser.add_argument('-c', '--continue_train', type=str2bool, nargs='?', default=False,
                        help='set continue train flag')

    parser.add_argument('--debug', action='store_true', help='need debug or not')


    args = parser.parse_args()

    print('\n\n{}\n\n'.format(args))
    tag = args.tag
    if tag == 'unknown_tag':
        tag = input('Enter log tag : ')
        print('\nSet log tag :"%s" ok !!\n' % tag)

    max_iter = args.max_iter

    weights = []
    if args.weights != '':
        weights = all.split(',') if args.weights == 'all' else args.weights.split(',')

    targets = []
    if args.targets != '':
        targets = args.targets.split(',')

    dataset_dir = config.cfg.PREPROCESSED_DATA_SETS_DIR

    use_raw = False
    train_n_val_dataset = []
    if use_raw:
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
    else:
        train_n_val_dataset = []

    data_splitter = TrainingValDataSplitter(train_n_val_dataset, use_raw)

    with BatchLoading(tags=data_splitter.training_tags, require_shuffle=True, random_num=np.random.randint(100),
                      is_flip=False) as training:
        with BatchLoading(tags=data_splitter.val_tags, queue_size=1, require_shuffle=True,random_num=666) as validation:
            train = mv3d.Trainer(train_set=training, validation_set=validation,
                                 pre_trained_weights=weights, train_targets=targets, log_tag=tag,
                                 continue_train=args.continue_train,
                                 fast_test_mode=True if max_iter == 1 else False, debug_mode=args.debug)
            train(max_iter=max_iter)
            # print(data_splitter.training_bags)
