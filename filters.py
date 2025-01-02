import numpy as np

class Filter:
    """
    Base class for filters.
    """
    def apply(self, image):
        """
        Apply the filter to the given image.
        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class ConvolutionFilter(Filter):
    """
    Generic class for convolution-based filters.
    """
    def __init__(self, mask):
        """
        Initialize the filter with a given mask.
        """
        self.mask = mask

    def apply(self, image):
        """
        Apply the convolution filter to the given image.
        """
        h, w = image.shape
        m, n = self.mask.shape
        pad_h, pad_w = m // 2, n // 2

        # Pad the image
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        filtered_image = np.zeros_like(image)

        # Perform convolution
        for i in range(h):
            for j in range(w):
                region = padded_image[i:i + m, j:j + n]
                filtered_image[i, j] = np.sum(region * self.mask)

        # Clip values to the valid range [0, 255]
        return np.clip(filtered_image, 0, 255).astype(np.uint8)

class HighPassFilter(ConvolutionFilter):
    """
    High-pass filter for sharpening an image.
    """
    def __init__(self):
        super().__init__(np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]]))

class LowPassFilter(ConvolutionFilter):
    """
    Low-pass filter for smoothing an image.
    """
    def __init__(self):
        super().__init__((1/6) * np.array([[0, 1, 0],
                                           [1, 2, 1],
                                           [0, 1, 0]]))

class MedianFilter(Filter):
    """
    Median filter for noise reduction.
    """
    def __init__(self, kernel_size):
        """
        Initialize the median filter with a given kernel size.
        """
        self.kernel_size = kernel_size

    def apply(self, image):
        """
        Apply the median filter to the given image.
        """
        h, w = image.shape
        pad = self.kernel_size // 2
        padded_image = np.pad(image, pad, mode='constant', constant_values=0)
        filtered_image = np.zeros_like(image)

        # Precompute kernel area
        kernel_area = self.kernel_size * self.kernel_size

        # Apply median filter
        for i in range(h):
            for j in range(w):
                # Get the flattened window (using the padding)
                region = padded_image[i:i + self.kernel_size, j:j + self.kernel_size].flatten()
                # Find the median using partition (much faster than sorting)
                filtered_image[i, j] = np.partition(region, kernel_area // 2)[kernel_area // 2]

        return filtered_image
