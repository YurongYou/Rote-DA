import os
import os.path as osp
import pickle
import sys

import hydra
import numpy as np
import scipy
from omegaconf import DictConfig, OmegaConf
from pyquaternion import Quaternion
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm

# from utils.pointcloud_utils import load_velo_scan, transform_points

from ithaca365.ithaca365 import Ithaca365
# from ithaca365.utils.data_io import load_velo_scan
# from ithaca365.utils.splits import create_splits_logs


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag

def display_args(args):
    eprint("========== ephemerality info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("=======================================")

def compute_stats_ithaca365(args):
    dataset = Ithaca365(version=args.data_paths.version, dataroot=args.data_paths.scan_path, verbose=True)
    only_accurate_localization = args.data_paths.only_accurate_localization
    # import ipdb; ipdb.set_trace()
    widths, lengths, heights = [], [], []
    for sample in tqdm(dataset.sample
                   if not only_accurate_localization else 
                   dataset.sample_with_accurate_localization):
    # for sample in tqdm(dataset.sample_with_accurate_localization[args.start_idx:args.end_idx]):
        sd_token = sample['data']['LIDAR_TOP']
        # sample_token = sample['token']  # save each according to sample token

        # segmentations = dataset.get_point_persistency_score(sd_token, num_histories=5, ranges=(0, 70))
        _, box_list, _ = dataset.get_sample_data(sd_token)
        for box in box_list:
            stats = box.wlh
            if box.name == "car":
                widths.append(stats[0])
                lengths.append(stats[1])
                heights.append(stats[2])
    widths = np.array(widths)
    lengths = np.array(lengths)
    heights = np.array(heights)
    print("width:", widths.mean(), widths.std())
    print("lengths:", lengths.mean(), lengths.std())
    print("heights:", heights.mean(), heights.std())

@hydra.main(config_path="configs/", config_name="p2_score.yaml")
def main(args: DictConfig):
    display_args(args)
    if args.dataset == "ithaca365":
        compute_stats_ithaca365(args)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
