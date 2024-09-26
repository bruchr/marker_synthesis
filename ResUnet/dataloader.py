import glob

import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset


class Dataset_own(Dataset):
    """Dataloader"""

    def __init__(self, configs, root_dir, transform=None):
        """
        Dataloader object which contains the files in a list and can load the data by their index
        :param root_dir: Path to the dataset containing an 'images' and a 'label_images' directory.
            :type root_dir: str
        :param label_type: Type of the label to use: binary, canny_adapted_2d, canny_adapted_3d, dilation_border and
            dilation_boundary.
            :type label_type: str
        :param shape: Shape of the training-subvolumes to create.
            :type label_type: list
        :param transform: Transforms/augmentations to apply.
            :type transform:
        """

        self.img_ids = sorted(glob.glob('{0}/patched_images/*.tif*'.format(root_dir)))
        self.transform = transform
        self.img_channel = configs['channel_input']
        self.label_channel = configs['channel_label']

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        raw_img = tiff.imread(self.img_ids[idx])
        img = np.expand_dims(raw_img[:,self.img_channel,...], axis=0)
        label = np.expand_dims(raw_img[:,self.label_channel,...], axis=0)

        assert img.min() < img.max(), "Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())

        sample = {'image': img, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample