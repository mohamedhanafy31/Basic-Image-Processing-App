import numpy as np
import matplotlib.pyplot as plt


class Thresholding:
    @staticmethod
    def calculate_average_threshold(image):
        """Calculate the average pixel intensity as a threshold."""
        if len(image.shape) > 2:
            raise ValueError("Input image must be grayscale")
        return int(np.mean(image))


class SimpleHalftoning:
    @staticmethod
    def apply_simple_halftoning(image, threshold):
        """Apply simple halftoning based on a threshold."""
        return np.where(image >= threshold, 255, 0).astype(np.uint8)


class AdvancedHalftoning:
    @staticmethod
    def error_diffusion_halftoning(image, threshold=128):
        """Perform error diffusion halftoning."""
        img_array = image.astype(np.float32)
        height, width = img_array.shape

        for i in range(height):
            for j in range(width):
                old_pixel = img_array[i, j]
                new_pixel = 255 if old_pixel >= threshold else 0
                img_array[i, j] = new_pixel

                error = old_pixel - new_pixel

                # Propagate error to neighbors
                if j + 1 < width:
                    img_array[i, j + 1] += error * 7 / 16
                if i + 1 < height and j > 0:
                    img_array[i + 1, j - 1] += error * 3 / 16
                if i + 1 < height:
                    img_array[i + 1, j] += error * 5 / 16
                if i + 1 < height and j + 1 < width:
                    img_array[i + 1, j + 1] += error * 1 / 16

        return np.clip(img_array, 0, 255).astype(np.uint8)


class Histogram:
    @staticmethod
    def plot_histogram(image):
        """Plot the histogram of a grayscale image."""
        histogram = np.zeros(256, dtype=int)
        height, width = image.shape

        for i in range(height):
            for j in range(width):
                pixel_value = image[i, j]
                histogram[pixel_value] += 1

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(256), histogram, color='black', linewidth=2)
        plt.title('Histogram of Grayscale Image')
        plt.xlabel('Pixel Intensity (0-255)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
        return histogram


class HistogramEqualization:
    @staticmethod
    def equalize(image):
        """Perform histogram equalization."""
        img_array = image.astype(np.float32)
        flat = img_array.flatten()

        hist, bins = np.histogram(flat, bins=256, range=[0, 256], density=True)
        cdf = hist.cumsum()
        cdf_normalized = cdf * (255 / cdf[-1])

        equalized_img_array = np.interp(flat, bins[:-1], cdf_normalized).reshape(img_array.shape)
        return np.clip(equalized_img_array, 0, 255).astype(np.uint8)