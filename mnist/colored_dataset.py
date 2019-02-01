import numpy as np
from PIL import Image
import random
import torch
from torchvision import datasets, transforms


class ColoredMNIST(datasets.MNIST):
    color_name_map = ["red", "green", "blue", "yellow", "magenta", "cyan", "purple", "lime", "orange", "white"]
    color_map = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                 np.array([1, 1, 0]), np.array([1, 0, 1]), np.array([0, 1, 1]),
                 np.array([0.57, 0.12, 0.71]), np.array([0.72, 0.96, 0.24]), np.array([0.96, 0.51, 0.19]),
                 np.array([1, 1, 1])]

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

        # Color image  TODO add logic to leave combination(s) out
        color_class = random.randrange(0, 9)

        img_array = img_array * self.color_map[color_class]
        img_array = img_array.astype("uint8")

        img = Image.fromarray(img_array)

        # Perform any non-color transformations
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return transforms.ToTensor()(img), torch.tensor([target.item(), color_class])
