import cv2
import numpy as np
from skimage.transform import resize


class Rescale():
    """Rescale the image, binary mask, and instance mask in a sample to a given size.

    Args:
        output_size (width, height) (tuple): Desired output size (width, height).
    """

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size  # (width, height)

    def __call__(self, sample):
        image, binary_mask, instance_mask = sample
        # Resizes the input image using bilinear interpolation
        image_resized = cv2.resize(image, self.output_size, interpolation=cv2.INTER_LINEAR)
        binary_mask_resized = cv2.resize(binary_mask, self.output_size, interpolation=cv2.INTER_NEAREST)
        instance_mask_resized = cv2.resize(instance_mask, self.output_size, interpolation=cv2.INTER_NEAREST)

        return image_resized, binary_mask_resized, instance_mask_resized

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, binary_mask, instance_mask = sample

        # Convert HWC to CHW and scale image pixels to [0,1]
        image = image.transpose((2, 0, 1))  
        image = image.astype(np.float32) / 255.0
        binary_mask = binary_mask.astype(np.int64)
        instance_mask = instance_mask.astype(np.int64)

        import torch
        return (torch.from_numpy(image),
                torch.from_numpy(binary_mask),
                torch.from_numpy(instance_mask))

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample