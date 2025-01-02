import numpy as np
import cv2
from scipy.signal import find_peaks


class ManualSegmentation:
    @staticmethod
    def segment(image, low_threshold, high_threshold, value=255):
        """Segment the image using manual thresholding."""
        segmented_image = np.zeros_like(image)
        segmented_image[(image >= low_threshold) & (image <= high_threshold)] = value
        return segmented_image


class HistogramPeakSegmentation:
    @staticmethod
    def segment(image, value=255):
        """Segment the image using histogram peak-based thresholding."""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        peak_indices = HistogramPeakSegmentation.find_histogram_peaks(hist)
        
        if len(peak_indices) < 2:
            raise ValueError("Less than two peaks found in the histogram.")
        
        low_threshold, high_threshold = HistogramPeakSegmentation.calculate_thresholds(peak_indices, hist)        
        segmented_image = np.zeros_like(image)
        segmented_image[(image >= low_threshold) & (image <= high_threshold)] = value
        
        return segmented_image, low_threshold, high_threshold
    
    @staticmethod
    def find_histogram_peaks(hist):
        """Find peaks in the histogram."""
        peaks, _ = find_peaks(hist, height=0)
        sorted_peaks = sorted(peaks, key=lambda x: hist[x], reverse=True)
        return sorted_peaks[:2]
    
    @staticmethod
    def calculate_thresholds(peak_indices, hist):
        """Calculate low and high thresholds based on histogram peaks."""
        peak1, peak2 = sorted(peak_indices)
        low_threshold = (peak1 + peak2) // 2
        high_threshold = peak2
        return low_threshold, high_threshold


class HistogramValleySegmentation:
    @staticmethod
    def segment(image, value=255):
        """Segment the image using histogram valley-based thresholding."""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        peak_indices = HistogramPeakSegmentation.find_histogram_peaks(hist)
        valley_point = HistogramValleySegmentation.find_valley_point(peak_indices, hist)
        low_threshold, high_threshold = HistogramValleySegmentation.valley_high_low(peak_indices, valley_point)        
        segmented_image = np.zeros_like(image)
        segmented_image[(image >= low_threshold) & (image <= high_threshold)] = value
        
        return segmented_image, low_threshold, high_threshold
    
    @staticmethod
    def find_valley_point(peaks_indices, hist):
        """Find the valley point between two peaks."""
        valley_point = 0
        min_valley = float('inf')
        start, end = peaks_indices
        for i in range(start, end + 1):
            if hist[i] < min_valley:
                min_valley = hist[i]
                valley_point = i
        return valley_point
    
    @staticmethod
    def valley_high_low(peak_indices, valley_point):
        """Calculate the low and high thresholds based on the valley point."""
        low_threshold = valley_point
        high_threshold = peak_indices[1]
        return low_threshold, high_threshold


class AdaptiveHistogramSegmentation:
    @staticmethod
    def segment(image, value=255):
        """Segment the image using adaptive histogram-based thresholding."""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        peak_indices = HistogramPeakSegmentation.find_histogram_peaks(hist)
        valley_point = HistogramValleySegmentation.find_valley_point(peak_indices, hist)
        low_threshold, high_threshold = HistogramValleySegmentation.valley_high_low(peak_indices, valley_point)        
        segmented_image = np.zeros_like(image)
        segmented_image[(image >= low_threshold) & (image <= high_threshold)] = value
        
        # Calculate object and background means
        background_mean, object_mean = AdaptiveHistogramSegmentation.calculate_means(segmented_image, image)
        new_peaks_indices = [int(background_mean), int(object_mean)]
        new_valley_point = HistogramValleySegmentation.find_valley_point(new_peaks_indices, hist)
        new_low_threshold, new_high_threshold = HistogramValleySegmentation.valley_high_low(new_peaks_indices, new_valley_point)
        
        final_segmented_image = np.zeros_like(image)
        final_segmented_image[(image >= new_low_threshold) & (image <= new_high_threshold)] = value
        
        return final_segmented_image, new_low_threshold, new_high_threshold
    
    @staticmethod
    def calculate_means(segmented_image, original_image):
        """Calculate the mean intensity of background and object regions."""
        object_pixels = original_image[segmented_image == 255]
        background_pixels = original_image[segmented_image == 0]
        
        object_mean = object_pixels.mean() if object_pixels.size > 0 else 0
        background_mean = background_pixels.mean() if background_pixels.size > 0 else 0
        
        return background_mean, object_mean
