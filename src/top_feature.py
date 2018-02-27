def top_feature(points, min_x, max_x, div_x, min_y, max_y, div_y, min_z, max_z, div_z):
    """
    extract
    1. height feature
    2. density feature
    3. integral image

    :param points: N x 4 (x, y, z, intensity), tensor
    :return:
    """

    # filter points out of image boundary

    # M: num of height slice, generate M x H x W tensor to hold height features

    # min_heights: a tensor min_heights arrange(0, M) * div_z

    #


    # generate H x W to hold density feature


    # generate H x W to hold integral image