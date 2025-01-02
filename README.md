# Advanced Image Processing Application 🚀🎨🖥️

Welcome to the **Advanced Image Processing Application**! 🌟

This project is a user-friendly, interactive **GUI application** designed to bring powerful image processing techniques to your fingertips. Built using **Python**, **CustomTkinter**, and **OpenCV**, this tool allows users to easily analyze, transform, and manipulate images with cutting-edge methods.

## Features

The application offers a range of tools for image preprocessing, edge detection, filtering, and segmentation. Key features include:

### ✅ **Preprocessing Tools**
- **Halftoning**: Converts images into dots to simulate a continuous-tone image using a limited number of colors.
- **Histogram Equalization**: Enhances image contrast by redistributing pixel intensity values.

### ✅ **Edge Detection**
- **Simple Methods**:
  - **Sobel**: Detects edges by computing gradients of the image intensity.
  - **Prewitt**: Similar to Sobel but uses a different convolution kernel.
  - **Kirsch**: Enhances edge detection by using eight convolution kernels.
- **Advanced Methods**:
  - **Difference of Gaussian (DoG) Convolution**: A technique to detect edges by subtracting two Gaussian blurred images.
  - **Contrast-based Edge Detection**: Extracts edges based on variations in contrast within the image.

### ✅ **Filtering Capabilities**
- **High-pass filters**: Removes low-frequency components and highlights high-frequency features like edges.
- **Low-pass filters**: Smoothens the image by removing high-frequency noise.
- **Median filters**: Reduces noise while preserving edges by replacing pixel values with the median value in a neighborhood.

### ✅ **Basic Operations**
- **Image Addition**: Combine two images by adding pixel values.
- **Image Subtraction**: Subtract pixel values of one image from another.
- **Image Inversion**: Inverts the pixel values, creating a negative of the image.

### ✅ **Segmentation**
- **Manual Segmentation**: Select regions manually to extract specific parts of the image.
- **Automated Segmentation**: Uses adaptive histogram-based methods for automatic segmentation of the image.

## Technologies Used

The following technologies were used to build this application:

- **Python**: The main programming language used for development.
- **CustomTkinter**: A Python library for creating sleek, modern, and responsive graphical user interfaces (GUIs).
- **OpenCV**: A powerful library for computer vision and image processing tasks.
- **NumPy**: Used for efficient numerical computations.
- **PIL (Pillow)**: A Python Imaging Library for opening, manipulating, and saving many different image file formats.

## Installation

To get started with the **Advanced Image Processing Application**, follow these steps:

1. **Clone the repository**:

```bash
git clone https://github.com/your-username/advanced-image-processing.git
cd advanced-image-processing
