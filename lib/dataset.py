import torch as th
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
import json
import os.path as path


class NNSimDataset(Dataset):

    def __init__(self, root):
        super().__init__()
        self.root = root
        with open(self.getpath("geometry_params.json")) as fp:
            self.geometry_params = json.load(fp)

    def __len__(self):
        # return len(self.geometry_params)
        return 8000

    def __getitem__(self, index):
        image = read_image(
            self.getpath("img", self.geometry_params[index]["filename"])
        )
        thickness = th.eye(16)[self.geometry_params[index]
                               ["thickness"]-1].float()
        spectrum = np.load(self.getpath("spectrums", "{}.npy".format(index)))
        return (image/255, thickness), np.array(spectrum, dtype=np.float32)

    def getpath(self, *x):
        return path.join(self.root, *x)


if __name__ == "__main__":
    dataset = NNSimDataset("dataset")
    (img, thickness), spectrum = dataset.__getitem__(0)
    print(img.shape, thickness, spectrum.shape)
