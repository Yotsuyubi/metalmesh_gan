import torch as th
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
import json
import os.path as path


class GANDataset(Dataset):

    def __init__(self, root):
        super().__init__()
        self.root = root
        with open(self.getpath("geometry_params.json")) as fp:
            self.geometry_params = json.load(fp)

    def __len__(self):
        return len(self.geometry_params)

    def __getitem__(self, index):
        image = read_image(
            self.getpath("imgs", self.geometry_params[index]["filename"])
        )
        return image/255

    def getpath(self, *x):
        return path.join(self.root, *x)


if __name__ == "__main__":
    dataset = GANDataset("dataset")
    img = dataset.__getitem__(0)
    print(img.shape)
