import cv2
import numpy as np

# Base Class
class ImageProcessor:
    def __init__(self, image):
        if isinstance(image, str):
            # Load the image from the file path
            self.image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image, np.ndarray):
            # Use the numpy array directly
            self.image = image
        else:
            raise TypeError("Unsupported image type. Provide a file path or numpy array.")

# Simple Edge Detection Class
class SimpleEdgeDetection(ImageProcessor):
    def apply_sobel(self, customThreshold=False, threshold= None):
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        gradient_x = cv2.filter2D(self.image, cv2.CV_32F, sobel_x)
        gradient_y = cv2.filter2D(self.image, cv2.CV_32F, sobel_y)
        magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if customThreshold:
            if threshold is not None:
                return (normalized > threshold).astype(np.uint8) * 255
        return normalized

    def apply_prewitt(self, customThreshold=False, threshold= None):
        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        gradient_x = cv2.filter2D(self.image, cv2.CV_32F, prewitt_x)
        gradient_y = cv2.filter2D(self.image, cv2.CV_32F, prewitt_y)
        magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if customThreshold:
            if threshold is not None:
                return (normalized > threshold).astype(np.uint8) * 255
        return normalized

    def apply_kirsch(self):
        directions = {1 : "N",
                      2 : "NW",
                      3 : "W",
                      4 : "SW",
                      5 : "S",
                      6 : "SE",
                      7 : "E",
                      8 : "NE"
                      }
        kernels = [
            np.array([[ 5,  5,  5],
                    [-3,  0, -3],
                    [-3, -3, -3]]),

            np.array([[ 5,  5, -3],
                    [ 5,  0, -3],
                    [-3, -3, -3]]),

            np.array([[ 5, -3, -3],
                    [ 5,  0, -3],
                    [ 5, -3, -3]]),

            np.array([[-3, -3, -3],
                    [ 5,  0, -3],
                    [ 5,  5, -3]]),

            np.array([[-3, -3, -3],
                    [-3,  0, -3],
                    [ 5,  5,  5]]),

            np.array([[-3, -3, -3],
                    [-3,  0,  5],
                    [-3,  5,  5]]),

            np.array([[-3, -3,  5],
                    [-3,  0,  5],
                    [-3, -3,  5]]),

            np.array([[-3,  5,  5],
                    [-3,  0,  5],
                    [-3, -3, -3]])
        ]
        edge_image = np.zeros_like(self.image, dtype=np.float32)
        total_responses = np.zeros(len(kernels))  # To store the sum of responses for each kernel

        for i, kernel in enumerate(kernels):
            convolved = cv2.filter2D(self.image, -1, kernel)
            edge_image = np.maximum(edge_image, convolved)
            
            # Sum responses for this direction
            total_responses[i] = np.sum(convolved)

        # Determine the best direction
        best_direction_index = np.argmax(total_responses) + 1  # Add 1 because directions are 1-indexed
        best_direction = directions[best_direction_index]

        return np.clip(edge_image, 0, 255).astype(np.uint8), best_direction

# Advanced Edge Detection Class
class AdvancedEdgeDetection(ImageProcessor):
    def homogeneity_operator(self, customThreshold=False, threshold=None):
        rows, cols = self.image.shape
        homogeneity_image = np.zeros_like(self.image)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center_pixel = self.image[i, j]
                neighbors = [
                    abs(center_pixel - self.image[i - 1, j - 1]),
                    abs(center_pixel - self.image[i - 1, j]),
                    abs(center_pixel - self.image[i - 1, j + 1]),
                    abs(center_pixel - self.image[i, j - 1]),
                    abs(center_pixel - self.image[i, j + 1]),
                    abs(center_pixel - self.image[i + 1, j - 1]),
                    abs(center_pixel - self.image[i + 1, j]),
                    abs(center_pixel - self.image[i + 1, j + 1]),
                ]
                homogeneity_image[i, j] = max(neighbors)

        # Normalize the image
        normalized = cv2.normalize(homogeneity_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply threshold if specified
        if customThreshold and threshold is not None:
            return (normalized > threshold).astype(np.uint8) * 255
        
        return normalized


    def difference_operator(self, customThreshold=False, threshold=None):
        rows, cols = self.image.shape
        difference_image = np.zeros_like(self.image)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                diff1 = abs(self.image[i - 1, j - 1] - self.image[i + 1, j + 1])
                diff2 = abs(self.image[i - 1, j + 1] - self.image[i + 1, j - 1])
                diff3 = abs(self.image[i, j - 1] - self.image[i, j + 1])
                diff4 = abs(self.image[i - 1, j] - self.image[i + 1, j])
                difference_image[i, j] = max(diff1, diff2, diff3, diff4)

        # Normalize the image
        normalized = cv2.normalize(difference_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply threshold if specified
        if customThreshold and threshold is not None:
            return (normalized > threshold).astype(np.uint8) * 255
        
        return normalized


    def dog_convolution(self, mask1, mask2):
        image7_7 = cv2.filter2D(self.image, -1, mask1)
        image9_9 = cv2.filter2D(self.image, -1, mask2)
        dog = image7_7 - image9_9
        return dog, image7_7, image9_9

    def contrast_based_edge_detection(self):
        edge_mask = np.array([[-1, 0, -1], [0, 4, 0], [-1, 0, -1]])
        smoothing_mask = np.ones((3, 3)) / 9
        image_float = self.image.astype(float)
        edge_output = cv2.filter2D(image_float, -1, edge_mask)
        average_output = cv2.filter2D(image_float, -1, smoothing_mask)
        average_output = np.maximum(average_output, 1e-10)  # Avoid division by zero
        contrast_edges = edge_output / average_output
        
        # Normalize and convert to uint8
        normalized = cv2.normalize(contrast_edges, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)


    def variance_operator(self):
        output = np.zeros_like(self.image, dtype=np.float32)
        rows, cols = self.image.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                neighborhood = self.image[i - 1:i + 2, j - 1:j + 2]
                mean = np.mean(neighborhood)
                variance = np.sum((neighborhood - mean) ** 2) / 9
                output[i, j] = variance
            # Normalize and convert to uint8
        normalized = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    def range_operator(self):
        output = np.zeros_like(self.image, dtype=np.float32)
        rows, cols = self.image.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                neighborhood = self.image[i - 1:i + 2, j - 1:j + 2]
                output[i, j] = np.max(neighborhood) - np.min(neighborhood)
        normalized = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)
