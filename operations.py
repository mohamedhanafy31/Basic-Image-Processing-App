import numpy as np
import cv2

class Operations:
    @staticmethod
    def add(image1, image2):
        """
        Add two images pixel by pixel, clipped to a maximum of 255.
        If images have different shapes, adjust by resizing or converting grayscale to color.
        """
        # Ensure images have the same dimensions
        if image1.shape != image2.shape:
            if len(image1.shape) == 2:  # Grayscale image
                # Convert grayscale to color by replicating channels
                image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
            if len(image2.shape) == 2:  # Grayscale image
                # Convert grayscale to color by replicating channels
                image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
            if image1.shape != image2.shape:
                # Resize image2 to match image1's dimensions
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

        # Initialize the added image with the same shape as image1
        added_image = np.zeros_like(image1, dtype=np.uint8)

        # Perform addition pixel by pixel
        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):
                # Perform addition on each channel individually for color images
                added_image[i, j, 0] = np.clip(image1[i, j, 0] + image2[i, j, 0], 0, 255)  # Blue
                added_image[i, j, 1] = np.clip(image1[i, j, 1] + image2[i, j, 1], 0, 255)  # Green
                added_image[i, j, 2] = np.clip(image1[i, j, 2] + image2[i, j, 2], 0, 255)  # Red

        return added_image

    @staticmethod
    def subtract(image1, image2):
        # Ensure images have the same dimensions
        if image1.shape != image2.shape:
            if len(image1.shape) == 2:  # Grayscale image
                # Convert grayscale to color by replicating channels
                image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

        # Initialize the subtracted image with the same shape as image1
        subtracted_image = np.zeros_like(image1, dtype=np.uint8)
        print(f"Shape of image1: {image1.shape}")
        print(f"Shape of image2: {image2.shape}")

        # Perform subtraction pixel by pixel
        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):
                # Perform subtraction on each channel individually for color images
                subtracted_image[i, j, 0] = np.clip(image1[i, j, 0] - image2[i, j, 0], 0, 255)  # Blue
                subtracted_image[i, j, 1] = np.clip(image1[i, j, 1] - image2[i, j, 1], 0, 255)  # Green
                subtracted_image[i, j, 2] = np.clip(image1[i, j, 2] - image2[i, j, 2], 0, 255)  # Red

        return subtracted_image
    
    @staticmethod
    def invert(image):
        """
        Invert the pixel values of the image (255 - pixel_value).
        """
        height, width = image.shape[:2]
        inverted_image = np.zeros_like(image, dtype=np.uint8)

        if len(image.shape) == 2:  # Grayscale image
            for i in range(height):
                for j in range(width):
                    inverted_image[i, j] = 255 - image[i, j]
        else:  # Color image (3 channels)
            for i in range(height):
                for j in range(width):
                    for c in range(3):  # Iterate over the 3 color channels (BGR)
                        inverted_image[i, j, c] = 255 - image[i, j, c]

        return inverted_image
