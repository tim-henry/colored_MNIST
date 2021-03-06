import numpy as np
from PIL import Image
import random
import torch
from torchvision import datasets, transforms


class LeftOutColoredMNIST(datasets.MNIST):
    # Color classes
    color_name_map = ["red", "green", "blue", "yellow", "magenta", "cyan", "purple", "lime", "orange", "white"]
    color_map = [np.array([1, 0.1, 0.1]), np.array([0.1, 1, 0.1]), np.array([0.1, 0.1, 1]),
                 np.array([1, 1, 0.1]), np.array([1, 0.1, 1]), np.array([0.1, 1, 1]),
                 np.array([0.57, 0.12, 0.71]), np.array([0.72, 0.96, 0.24]), np.array([0.96, 0.51, 0.19]),
                 np.array([1, 1, 1])]

    # Gaussian noise arguments
    mu = 0
    sigma = 200

    # pct_to_keep: percentage of possible combinations to keep between 0 and 1, rounded down to nearest multiple of 0.2
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, pct_to_keep=1, color_indices=np.arange(10)):
        super().__init__(root, train, transform, target_transform, download)
        pct = pct_to_keep * 10
        self.max_left_dist = int(pct / 2)
        self.max_right_dist = int(pct / 2) if pct % 2 == 0 else int(pct / 2) + 1
        self.held_out = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 0), (6, 1), (7, 2), (8, 3), (9, 4)]
        self.control = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
        self.color_indices = color_indices

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
        lower_bound = number_class - self.max_left_dist
        upper_bound = number_class + self.max_right_dist

        color_class = random.randrange(lower_bound, upper_bound) % 10

        img_array = img_array * self.color_map[self.color_indices[color_class]]

        # Add Gaussian noise
        noise = np.reshape(np.random.normal(self.mu, self.sigma, img_array.size), img_array.shape)
        mask = (img_array != 0).astype("uint8")
        img_array = img_array + np.multiply(mask, noise)
        img_array = np.clip(img_array, 0, 255)
        img_array = img_array.astype("uint8")

        img = Image.fromarray(img_array)

        # Perform any non-color transformations
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return transforms.ToTensor()(img), torch.tensor([target.item(), color_class])
