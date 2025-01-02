import os
import numpy as np
import cv2
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.simpledialog as simpledialog
from edge_detection import SimpleEdgeDetection, AdvancedEdgeDetection
from imagePreprocessing import Thresholding, SimpleHalftoning, AdvancedHalftoning, Histogram, HistogramEqualization
from filters import HighPassFilter, LowPassFilter, MedianFilter
from operations import Operations
from segemetaion import ManualSegmentation, AdaptiveHistogramSegmentation, HistogramPeakSegmentation, HistogramValleySegmentation

class ImageProcessingApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Initialize image history
        self.history = []

        # Configure window
        self.title("Advanced Image Processing")
        self.geometry("1400x700")
        
        # Configure grid layout
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=4)
        self.grid_rowconfigure(2, weight=4)

        
        # Titles above images
        self.current_image_title_label = ctk.CTkLabel(self, text="WARNING! Note that any operation rather than the preprocessing oprtations that will be applied on the grayscaly level of the uploaded image", font=("Roboto", 15, "bold"), text_color= "yellow")
        self.current_image_title_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew")  # Title above current image
        
        # Image display area
        self.image_label = ctk.CTkLabel(self, text="Current Image", font=("Arial", 16))
        self.image_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.original_image_label = ctk.CTkLabel(self, text="Original Image", font=("Arial", 16))
        self.original_image_label.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        
        # Create scrollable frame for sidebar
        self.sidebar = ctk.CTkScrollableFrame(self, width=200, height=500, corner_radius=0, fg_color="transparent")
        self.sidebar.grid(row=1, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        # Upload button
        self.upload_button = ctk.CTkButton(self.sidebar, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10, padx=10, fill='x')
        
        # Create operation sections
        self.create_preprocrssing_section()
        self.create_edge_detection_section()
        self.create_filters_section()
        self.create_basic_operations_section()
        self.create_segmentation_section()
        

        # Current image state
        self.current_image = None
        self.original_image = None
        self.grayscale_image = None

    def upload_image(self):
        filepath = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.JPEG")]
        )
        
        if filepath:
            try:
                # Verify file exists and format is valid
                from PIL import Image
                with Image.open(filepath) as img:
                    print(f"Selected image format: {img.format}")
                
                # Read the image
                self.original_image = cv2.imread(filepath)
                if self.original_image is None:
                    raise ValueError("Failed to load image. File may be corrupted or unsupported.")
                
                # Automatically convert to grayscale
                self.grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                self.current_image = self.grayscale_image
                if len(self.current_image.shape) > 2:
                    grayscale = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
                else:
                    grayscale = self.current_image
                
                # Calculate the threshold
                self.threshold_value = Thresholding.calculate_average_threshold(grayscale)
                
                # Display both images
                self.display_image(self.current_image, self.original_image)
                
                # Inform the user
                messagebox.showinfo("Image Uploaded", 
                                    f"The image has been successfully uploaded and converted to grayscale.\n Average Threshold value is {self.threshold_value}")
            
            except Exception as e:
                messagebox.showerror("Error", f"Could not open image: {str(e)}")

    def display_image(self, current_image=None, original_image=None):
        if current_image is not None:
            # Convert current image to RGB
            if len(current_image.shape) == 2:
                rgb_current_image = cv2.cvtColor(current_image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
    
            # Convert to PhotoImage
            current_photo = ImageTk.PhotoImage(Image.fromarray(rgb_current_image))
            
            # Update current image label and store a reference to avoid garbage collection
            self.image_label.configure(image=current_photo, text="")
            self.image_label.image = current_photo  # Store the reference here
    
        if original_image is not None:
            # Convert original image to RGB
            if len(original_image.shape) == 2:
                rgb_original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
            # Convert to PhotoImage
            original_photo = ImageTk.PhotoImage(Image.fromarray(rgb_original_image))
            
            # Update original image label and store a reference to avoid garbage collection
            self.original_image_label.configure(image=original_photo, text="")
            self.original_image_label.image = original_photo  # Store the reference here

    def create_preprocrssing_section(self):
        # Basic Operations Section
        basic_section = ctk.CTkLabel(self.sidebar, text="Preprocessing Operations", font=("Arial", 16, "bold"))
        basic_section.pack(pady=(10, 5))
        
        # Thresholding Buttons
        threshold_frame = ctk.CTkFrame(self.sidebar)
        threshold_frame.pack(pady=5, padx=10, fill='x')
        
        ctk.CTkLabel(threshold_frame, text="Halftoning:").pack()
        
        simple_threshold_btn = ctk.CTkButton(threshold_frame, text="Simple Halftone", command=self.apply_simple_halftone)
        simple_threshold_btn.pack(pady=5, fill='x')
        
        error_diffusion_btn = ctk.CTkButton(threshold_frame, text="Error Diffusion (Advanced Halftone)", command=self.apply_error_diffusion)
        error_diffusion_btn.pack(pady=5, fill='x')
        
        # Histogram Operations
        histogram_frame = ctk.CTkFrame(self.sidebar)
        histogram_frame.pack(pady=5, padx=10, fill='x')
        
        ctk.CTkLabel(histogram_frame, text="Histogram:").pack()
        
        generate_hist_btn = ctk.CTkButton(histogram_frame, text="Generate Histogram", command=self.generate_histogram)
        generate_hist_btn.pack(pady=5, fill='x')
        
        equalize_hist_btn = ctk.CTkButton(histogram_frame, text="Histogram Equalization", command=self.equalize_histogram)
        equalize_hist_btn.pack(pady=5, fill='x')

    def create_edge_detection_section(self):
        # Simple Edge Detection Section
        simple_edge_section = ctk.CTkLabel(self.sidebar, text="Simple Edge Detection", font=("Arial", 16, "bold"))
        simple_edge_section.pack(pady=(10, 5))

        # Buttons for Simple Edge Detection
        simple_edge_buttons = [
            ("Sobel Operator", self.apply_sobel),
            ("Prewitt Operator", self.apply_prewitt),
            ("Kirsch Operator", self.apply_kirsch)
        ]

        for name, command in simple_edge_buttons:
            btn = ctk.CTkButton(self.sidebar, text=name, command=command)
            btn.pack(pady=5, padx=10, fill='x')

        # Advanced Edge Detection Section
        advanced_edge_section = ctk.CTkLabel(self.sidebar, text="Advanced Edge Detection", font=("Arial", 16, "bold"))
        advanced_edge_section.pack(pady=(10, 5))

        advanced_edge_buttons = [
            ("Homogeneity Operator", self.apply_homogeneity_operator),
            ("Difference Operator", self.apply_difference_operator),
            ("DoG Convolution", self.apply_dog_convolution),
            ("Contrast-based Edges", self.apply_contrast_based_edges),
            ("Variance Operator", self.apply_variance_operator),
            ("Range Operator", self.apply_range_operator)
        ]

        for name, command in advanced_edge_buttons:
            btn = ctk.CTkButton(self.sidebar, text=name, command=command)
            btn.pack(pady=5, padx=10, fill='x')

    def create_filters_section(self):
        # Filters Section
        filters_section = ctk.CTkLabel(self.sidebar, text="Filters", font=("Arial", 16, "bold"))
        filters_section.pack(pady=(10, 5))
        
        # Filter Buttons
        filters_buttons = [
            ("High Pass Filter", self.apply_high_pass_filter),
            ("Low Pass Filter", self.apply_low_pass_filter),
            ("Median Filter", self.apply_median_filter)
        ]
        
        for name, command in filters_buttons:
            btn = ctk.CTkButton(self.sidebar, text=name, command=command)
            btn.pack(pady=5, padx=10, fill='x')

    def create_basic_operations_section(self):
        # Filters Section
        filters_section = ctk.CTkLabel(self.sidebar, text="Basic Operations Section", font=("Arial", 16, "bold"))
        filters_section.pack(pady=(10, 5))
        
        # Filter Buttons
        filters_buttons = [
            ("Add", self.add_image),
            ("Subtract", self.subtract_image),
            ("Invert", self.invert_image)
        ]
        
        for name, command in filters_buttons:
            btn = ctk.CTkButton(self.sidebar, text=name, command=command)
            btn.pack(pady=5, padx=10, fill='x')

    def create_segmentation_section(self):
        # Filters Section
        filters_section = ctk.CTkLabel(self.sidebar, text="Segmentaion Section", font=("Arial", 16, "bold"))
        filters_section.pack(pady=(10, 5))
        
        # Filter Buttons
        filters_buttons = [
            ("Manual Segmentation", self.show_segmentation_popup),
            ("Histogram Peak Segmentation", self.apply_histogram_peak_segmentation),
            ("Histogram Valley Segmentation", self.apply_histogram_valley_segmentation),
            ("Adaptive Histogram Segmentation", self.apply_adaptive_histogram_segmentation)
        ]
        
        for name, command in filters_buttons:
            btn = ctk.CTkButton(self.sidebar, text=name, command=command)
            btn.pack(pady=5, padx=10, fill='x')

    def apply_simple_halftone(self):
        if self.grayscale_image is not None:
            # Ensure image is grayscale
            if len(self.grayscale_image.shape) > 2:
                grayscale = cv2.cvtColor(self.grayscale_image, cv2.COLOR_BGR2GRAY)
            else:
                grayscale = self.grayscale_image
            
            
            # Apply simple halftoning
            thresholded = SimpleHalftoning.apply_simple_halftoning(grayscale, self.threshold_value)
            
            # Update current image and display
            self.current_image = thresholded
            self.display_image(thresholded)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")


    def apply_error_diffusion(self):
        if self.grayscale_image is not None:
            # Ensure image is grayscale
            if len(self.grayscale_image.shape) > 2:
                grayscale = cv2.cvtColor(self.grayscale_image, cv2.COLOR_BGR2GRAY)
            else:
                grayscale = self.grayscale_image
            
            error_diffused = AdvancedHalftoning.error_diffusion_halftoning(grayscale, self.threshold_value)
            
            self.current_image = error_diffused
            self.display_image(error_diffused)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")


    def generate_histogram(self):
        if self.current_image is not None:
            # Ensure image is grayscale
            if len(self.current_image.shape) > 2:
                grayscale = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                grayscale = self.current_image
            
            Histogram.plot_histogram(grayscale)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")


    def equalize_histogram(self):
        if self.current_image is not None:
            # Ensure image is grayscale
            if len(self.current_image.shape) > 2:
                grayscale = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                grayscale = self.current_image
            
            # Initialize the HistogramOperations class
            histogram_op = HistogramEqualization()
            
            # Perform histogram equalization
            equalized = histogram_op.equalize(grayscale)
            
            # Update the current image and display
            self.current_image = equalized
            self.grayscale_image = equalized
            self.display_image(equalized)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")


    def apply_sobel(self):
        if self.original_image is not None:
            def sobel_with_params(custom_threshold, threshold):
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                edge_detector = SimpleEdgeDetection(gray)
                sobel_edges = edge_detector.apply_sobel(customThreshold=custom_threshold, threshold=threshold)
                self.current_image = sobel_edges
                self.add_to_history(sobel_edges)
                self.display_image(sobel_edges)
    
            self.get_threshold_popup(sobel_with_params)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")
    
    def apply_prewitt(self):
        if self.original_image is not None:
            def prewitt_with_params(custom_threshold, threshold):
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                edge_detector = SimpleEdgeDetection(gray)
                prewitt_edges = edge_detector.apply_prewitt(customThreshold=custom_threshold, threshold=threshold)
                self.current_image = prewitt_edges
                self.add_to_history(prewitt_edges)
                self.display_image(prewitt_edges)
    
            self.get_threshold_popup(prewitt_with_params)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")



    def apply_kirsch(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            edge_detector = SimpleEdgeDetection(gray)
            kirsch_edges, bestDirection = edge_detector.apply_kirsch()
            self.show_direction_popup(bestDirection)
            self.current_image = kirsch_edges  # Update the current image for display
            self.add_to_history(kirsch_edges)  # Save the result in the history
            self.display_image(kirsch_edges)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")

    def apply_homogeneity_operator(self):
        if self.original_image is not None:
            def homogeneity_with_params(custom_threshold, threshold):
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                edge_detector = AdvancedEdgeDetection(gray)
                homogeneity_edges = edge_detector.homogeneity_operator()

                if custom_threshold and threshold is not None:
                    homogeneity_edges = (homogeneity_edges > threshold).astype(np.uint8) * 255

                self.current_image = homogeneity_edges
                self.add_to_history(homogeneity_edges)
                self.display_image(homogeneity_edges)

            self.get_threshold_popup(homogeneity_with_params)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")


    def apply_difference_operator(self):
        if self.original_image is not None:
            def difference_with_params(custom_threshold, threshold):
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                edge_detector = AdvancedEdgeDetection(gray)
                difference_edges = edge_detector.difference_operator()

                if custom_threshold and threshold is not None:
                    difference_edges = (difference_edges > threshold).astype(np.uint8) * 255

                self.current_image = difference_edges
                self.add_to_history(difference_edges)
                self.display_image(difference_edges)

            self.get_threshold_popup(difference_with_params)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")


    class CustomPopup(ctk.CTkToplevel):
        def __init__(self, parent, callback):
            super().__init__(parent)
            self.callback = callback
            self.selected_option = ctk.StringVar(value="DoG")

            self.title("Choose Result")
            self.geometry("300x200")
            self.configure(padx=20, pady=20)

            ctk.CTkLabel(self, text="Select the result to display:", font=("Arial", 14)).pack(pady=(0, 10))

            # Radio buttons for each option
            options = ["DoG", "7x7 Mask", "9x9 Mask"]
            for option in options:
                ctk.CTkRadioButton(self, text=option, variable=self.selected_option, value=option).pack(anchor="w")

            # Confirm button
            ctk.CTkButton(self, text="Confirm", command=self.confirm_selection).pack(pady=(20, 0))

        def confirm_selection(self):
            self.callback(self.selected_option.get())
            self.destroy()


    class CustomPopup(ctk.CTkToplevel):
        def __init__(self, parent, callback):
            super().__init__(parent)
            self.callback = callback
            self.selected_option = ctk.StringVar(value="DoG")

            self.title("Choose Result")
            self.geometry("300x200")
            self.configure(padx=20, pady=20)

            ctk.CTkLabel(self, text="Select the result to display:", font=("Arial", 14)).pack(pady=(0, 10))

            # Radio buttons for each option
            options = ["DoG", "7x7 Mask", "9x9 Mask"]
            for option in options:
                ctk.CTkRadioButton(self, text=option, variable=self.selected_option, value=option).pack(anchor="w")

            # Confirm button
            ctk.CTkButton(self, text="Confirm", command=self.confirm_selection).pack(pady=(20, 0))

        def confirm_selection(self):
            self.callback(self.selected_option.get())
            self.destroy()


    def apply_dog_convolution(self):
        if self.original_image is not None:
            # Convert the image to grayscale
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            edge_detector = AdvancedEdgeDetection(gray)

            # Define masks
            mask_7x7 = np.array([
                [0, 0, -1, -1, -1, 0, 0],
                [0, -2, -3, -3, -3, -2, 0],
                [-1, -3, 5, 5, 5, -3, -1],
                [-1, -3, 5, 16, 5, -3, -1],
                [-1, -3, 5, 5, 5, -3, -1],
                [0, -2, -3, -3, -3, -2, 0],
                [0, 0, -1, -1, -1, 0, 0]
            ], dtype=np.float32)

            mask_9x9 = np.array([
                [0, 0, 0, -1, -1, -1, 0, 0, 0],
                [0, -2, -3, -3, -3, -3, -3, -2, 0],
                [0, -3, -2, -1, -1, -1, -2, -3, 0],
                [-1, -3, -1, 9, 9, 9, -1, -3, -1],
                [-1, -3, -1, 9, 19, 9, -1, -3, -1],
                [-1, -3, -1, 9, 9, 9, -1, -3, -1],
                [0, -3, -2, -1, -1, -1, -2, -3, 0],
                [0, -2, -3, -3, -3, -3, -3, -2, 0],
                [0, 0, 0, -1, -1, -1, 0, 0, 0]
            ], dtype=np.float32)

            # Perform DoG convolution
            dog, result_7x7, result_9x9 = edge_detector.dog_convolution(mask_7x7, mask_9x9)

            # Callback function for result selection
            def handle_result_selection(choice):
                if choice == "DoG":
                    selected_result = dog
                elif choice == "7x7 Mask":
                    selected_result = result_7x7
                elif choice == "9x9 Mask":
                    selected_result = result_9x9
                else:
                    messagebox.showerror("Error", "Invalid choice! Please try again.")
                    return

                # Update current image and display the selected result
                self.current_image = selected_result
                self.add_to_history(selected_result)
                self.display_image(selected_result)

            # Show the custom popup
            popup = self.CustomPopup(self.master, handle_result_selection)
            popup.grab_set()
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")

    def apply_contrast_based_edges(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            edge_detector = AdvancedEdgeDetection(gray)
            contrast_edges = edge_detector.contrast_based_edge_detection()
            self.current_image = contrast_edges
            self.add_to_history(contrast_edges)
            self.display_image(contrast_edges)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")

    def apply_variance_operator(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            edge_detector = AdvancedEdgeDetection(gray)
            variance_edges = edge_detector.variance_operator()
            self.current_image = variance_edges
            self.add_to_history(variance_edges)
            self.display_image(variance_edges)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")

    def apply_range_operator(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            edge_detector = AdvancedEdgeDetection(gray)
            range_edges = edge_detector.range_operator()
            self.current_image = range_edges
            self.add_to_history(range_edges)
            self.display_image(range_edges)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")

    def apply_high_pass_filter(self):
        if self.original_image is not None:
            # Ensure image is grayscale
            if len(self.original_image.shape) > 2:
                grayscale = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            else:
                grayscale = self.original_image
            
            # Create and apply high-pass filter
            high_pass = HighPassFilter()
            filtered_image = high_pass.apply(grayscale)
            
            # Update current image and display
            self.current_image = filtered_image
            self.display_image(filtered_image)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")

    def apply_low_pass_filter(self):
        if self.original_image is not None:
            # Ensure image is grayscale
            if len(self.original_image.shape) > 2:
                grayscale = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            else:
                grayscale = self.original_image
            
            # Create and apply low-pass filter
            low_pass = LowPassFilter()
            filtered_image = low_pass.apply(grayscale)
            
            # Update current image and display
            self.current_image = filtered_image
            self.display_image(filtered_image)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")

    def apply_median_filter(self):
        if self.original_image is not None:
            # Ensure image is grayscale
            if len(self.original_image.shape) > 2:
                grayscale = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            else:
                grayscale = self.original_image
            
            # Create and apply median filter (3x3 kernel)
            median = MedianFilter(kernel_size=3)
            filtered_image = median.apply(grayscale)
            
            # Update current image and display
            self.current_image = filtered_image
            self.display_image(filtered_image)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")

    def add_image(self):
        if self.grayscale_image is not None:

            # Add image to its copy
            added_image = Operations.add(self.current_image, self.original_image)
            self.current_image = added_image
            self.display_image(added_image)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")

    def subtract_image(self):
        if self.grayscale_image is not None:
            # Subtract image from its copy
            print(f"Shape of current_image: {self.current_image.shape}")
            print(f"Shape of original_image: {self.original_image.shape}")

            subtracted_image = Operations.subtract(self.current_image, self.original_image)
            self.current_image = subtracted_image
            self.display_image(subtracted_image)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")

    def invert_image(self):
        if self.grayscale_image is not None:
            # Invert the image
            inverted_image = Operations.invert(self.grayscale_image)
            self.current_image = inverted_image
            self.display_image(inverted_image)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")


    # Apply the segmentation operations
    def apply_manual_segmentation(self, low_threshold, high_threshold):
        if self.grayscale_image is not None:
            # Apply manual segmentation
            segmented_image = ManualSegmentation.segment(self.grayscale_image, low_threshold, high_threshold)
            self.current_image = segmented_image
            self.display_image(segmented_image)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")


    def apply_histogram_peak_segmentation(self):
        if self.grayscale_image is not None:
            try:
                # Apply histogram peak-based segmentation and get the thresholds
                segmented_image, low_threshold, high_threshold = HistogramPeakSegmentation.segment(self.grayscale_image)
                
                # Show the thresholds in a popup
                self.show_thresholds_popup(low_threshold, high_threshold)
                
                # Update the current image and display the segmented image
                self.current_image = segmented_image
                self.display_image(segmented_image)
            except ValueError as e:
                messagebox.showerror("Error", f"Segmentation error: {e}")
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")

    def apply_histogram_valley_segmentation(self):
        if self.grayscale_image is not None:
            # Apply histogram valley-based segmentation and get the thresholds
            segmented_image, low_threshold, high_threshold = HistogramValleySegmentation.segment(self.grayscale_image)
            
            # Show the thresholds in a popup
            self.show_thresholds_popup(low_threshold, high_threshold)
            
            # Update the current image and display the segmented image
            self.current_image = segmented_image
            self.display_image(segmented_image)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")

    def apply_adaptive_histogram_segmentation(self):
        if self.grayscale_image is not None:
            # Apply adaptive histogram-based segmentation and get the thresholds
            segmented_image, low_threshold, high_threshold = AdaptiveHistogramSegmentation.segment(self.grayscale_image)
            
            # Show the thresholds in a popup
            self.show_thresholds_popup(low_threshold, high_threshold)
            
            # Update the current image and display the segmented image
            self.current_image = segmented_image
            self.display_image(segmented_image)
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")

    def get_threshold_popup(self, edge_detection_function):
        # Create popup window
        popup = ctk.CTkToplevel(self)
        popup.title("Threshold Selection")
        popup.geometry("400x200")
    
        # Make the popup window transient (it will stay on top of the main window)
        popup.transient(self)  # Make popup a child of the main window
        popup.grab_set()  # Ensure the popup is modal (blocks interaction with other windows)
        
        # Center the popup window on the main window
        window_width = 400
        window_height = 200
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        position_top = int(screen_height / 2 - window_height / 2)
        position_right = int(screen_width / 2 - window_width / 2)
        
        popup.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')
        
        # Question label
        question_label = ctk.CTkLabel(popup, text="Do you want to apply a threshold?", font=("Arial", 14))
        question_label.pack(pady=10)
    
        # Threshold slider
        threshold_var = ctk.DoubleVar(value=self.threshold_value)  # Default threshold value
        threshold_slider = ctk.CTkSlider(popup, from_=0, to=255, variable=threshold_var, state="disabled",  number_of_steps=256, 
                                         command=lambda val: question_label.configure(text=f"Threshold Value: {int(float(val))}"))
        threshold_slider.pack(pady=10, fill="x", padx=20)
    
        # Option to enable slider
        def toggle_slider():
            state = "normal" if enable_threshold.get() else "disabled"
            threshold_slider.configure(state=state)
    
        enable_threshold = ctk.BooleanVar(value=False)
        enable_threshold_checkbox = ctk.CTkCheckBox(
            popup,
            text="Apply Threshold",
            variable=enable_threshold,
            command=toggle_slider
        )
        enable_threshold_checkbox.pack(pady=10)
    
        # Submit button
        def on_submit():
            apply_threshold = enable_threshold.get()
            threshold_value = threshold_var.get() if apply_threshold else None
            popup.destroy()
            # Call the edge detection function with the parameters
            edge_detection_function(custom_threshold=apply_threshold, threshold=threshold_value)
    
        submit_button = ctk.CTkButton(popup, text="Submit", command=on_submit)
        submit_button.pack(pady=10)

    def show_segmentation_popup(self):
        # Create a new top-level window (popup) with customtkinter
        popup = ctk.CTkToplevel(self)
        popup.title("Manual Segmentation Thresholds")

        # Make the popup modal (prevents interaction with the main window until closed)
        popup.grab_set()

        # Add labels and entry fields for low and high thresholds using CTk widgets
        low_threshold_label = ctk.CTkLabel(popup, text="Low Threshold:")
        low_threshold_label.grid(row=0, column=0, padx=10, pady=10)
        low_threshold_entry = ctk.CTkEntry(popup)
        low_threshold_entry.grid(row=0, column=1, padx=10, pady=10)

        high_threshold_label = ctk.CTkLabel(popup, text="High Threshold:")
        high_threshold_label.grid(row=1, column=0, padx=10, pady=10)
        high_threshold_entry = ctk.CTkEntry(popup)
        high_threshold_entry.grid(row=1, column=1, padx=10, pady=10)

        def apply_thresholds():
            try:
                low_threshold = int(low_threshold_entry.get())
                high_threshold = int(high_threshold_entry.get())
                
                # Validate threshold values
                if low_threshold < 0 or low_threshold > 255 or high_threshold < 0 or high_threshold > 255:
                    raise ValueError("Threshold values must be between 0 and 255.")
                
                if low_threshold >= high_threshold:
                    messagebox.showerror("Invalid Thresholds", "Low Threshold must be less than High Threshold.")
                    return
                
                # Apply the manual segmentation with the user-defined thresholds
                self.apply_manual_segmentation(low_threshold, high_threshold)
                popup.destroy()  # Close the popup after applying

            except ValueError as e:
                messagebox.showerror("Invalid input", f"Please enter valid integer values for thresholds.\n{str(e)}")

        # Add Apply and Cancel buttons using CTk widgets
        apply_button = ctk.CTkButton(popup, text="Apply", command=apply_thresholds)
        apply_button.grid(row=2, column=0, padx=10, pady=10)
        cancel_button = ctk.CTkButton(popup, text="Cancel", command=popup.destroy)
        cancel_button.grid(row=2, column=1, padx=10, pady=10)

        # Bring the popup to the front
        popup.lift()

    def show_thresholds_popup(self, low_threshold, high_threshold):
        # Create a new top-level window (popup) with customtkinter
        popup = ctk.CTkToplevel(self)
        popup.title("Applied Thresholds")

        # Make the popup modal
        popup.grab_set()


        # Add labels to display the thresholds
        title_label = ctk.CTkLabel(popup, text=f"The Applied Threshold", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, padx=10, pady=10)

        # Add labels to display the thresholds
        low_threshold_label = ctk.CTkLabel(popup, text=f"Low Threshold: {low_threshold}")
        low_threshold_label.grid(row=1, column=0, padx=10, pady=10)
        
        high_threshold_label = ctk.CTkLabel(popup, text=f"High Threshold: {high_threshold}")
        high_threshold_label.grid(row=2, column=0, padx=10, pady=10)

        # Add OK button to close the popup
        ok_button = ctk.CTkButton(popup, text="OK", command=popup.destroy)
        ok_button.grid(row=3, column=0, padx=20, pady=20)

        # Bring the popup to the front
        popup.lift()

    def show_direction_popup(self, directionResult):
        # Create a new top-level window (popup) with customtkinter
        popup = ctk.CTkToplevel(self)
        popup.title("Applied Thresholds")

        # Make the popup modal
        popup.grab_set()


        # Add labels to display the thresholds
        title_label = ctk.CTkLabel(popup, text=f"The Best direction is {directionResult}", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, padx=10, pady=10)

        # Add OK button to close the popup
        ok_button = ctk.CTkButton(popup, text="OK", command=popup.destroy)
        ok_button.grid(row=3, column=0, padx=20, pady=20)

        # Bring the popup to the front
        popup.lift()
    def add_to_history(self, image):
        # Add the processed image to the history list
        self.history.append(image)
        print(f"History length: {len(self.history)}")

# Main function remains the same
def main():
    # Set appearance mode and color theme
    ctk.set_appearance_mode("dark")  # Modes: system (default), light, dark
    ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green
    
    app = ImageProcessingApp()
    app.mainloop()


if __name__ == "__main__":
    main()