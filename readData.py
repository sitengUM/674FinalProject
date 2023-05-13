import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)



def get_data(path):
    print("get_data called {}".format(path))

    pc = pd.read_csv(path,
                     header=None,
                     delim_whitespace=True,
                     dtype=np.float32).values

    points = pc[:, 0:3]
    feat = pc[:, [4, 5, 6]]
    intensity = pc[:, 3]

    points = np.array(points, dtype=np.float32)
    feat = np.array(feat, dtype=np.float32)
    intensity = np.array(intensity, dtype=np.float32)

    labels = pd.read_csv(path.replace(".txt", ".labels"),
                         header=None,
                         delim_whitespace=True,
                         dtype=np.int32).values
    labels = np.array(labels, dtype=np.int32).reshape((-1,))

    data = {
        'point': points,
        'feat': feat,
        'intensity': intensity,
        'label': labels
    }

    return data

if __name__ == '__main__':
    train_data = get_data("input/point-cloud-segmentation/train/bildstein_station3_xyz_intensity_rgb.txt")
    val_data =  get_data("input/point-cloud-segmentation/val/bildstein_station3_xyz_intensity_rgb.txt")


