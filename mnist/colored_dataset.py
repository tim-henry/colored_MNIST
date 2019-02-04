import numpy as np
from PIL import Image
import random
import torch
from torchvision import datasets, transforms


class LeftOutColoredMNIST(datasets.MNIST):
    color_name_map = ["red", "green", "blue", "yellow", "magenta", "cyan", "purple", "lime", "orange", "white"]
    color_map = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                 np.array([1, 1, 0]), np.array([1, 0, 1]), np.array([0, 1, 1]),
                 np.array([0.57, 0.12, 0.71]), np.array([0.72, 0.96, 0.24]), np.array([0.96, 0.51, 0.19]),
                 np.array([1, 1, 1])]

    # pct_to_keep: percentage of possible combinations to keep between 0 and 1, rounded down to nearest multiple of 0.2
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, pct_to_keep=1):
        super().__init__(root, train, transform, target_transform, download)

        self.max_dist = int((pct_to_keep * 10) / 2)

        self.left_out = []
        for i in range(10):
            leave_out = {k for k in range(10)}

            lower_bound = i - self.max_dist
            upper_bound = i + self.max_dist
            for j in range(lower_bound, upper_bound):
                leave_out.remove(j % 10)

            for j in leave_out:
                self.left_out.append((i, j))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # Put grayscale image in RGB space
        img_array = np.stack((img.numpy(),) * 3, axis=-1)

        # Color image
        number_class = target.item()
        lower_bound = number_class - self.max_dist
        upper_bound = number_class + self.max_dist

        color_class = random.randrange(lower_bound, upper_bound) % 10

        img_array = img_array * self.color_map[color_class]
        img_array = img_array.astype("uint8")

        img = Image.fromarray(img_array)

        # Perform any non-color transformations
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return transforms.ToTensor()(img), torch.tensor([target.item(), color_class])
