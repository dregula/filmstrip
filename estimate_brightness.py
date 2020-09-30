# estimate_brightness

import numpy as np
import cv2
from blur_detection import BlurDetection


# TODO: change threshold_method to string, then eval to be consistent
class EstimateBrightness:
    def __init__(self, threshold_method: str = "percent_pixels_below_threshold", cv2_threshold_method: int = cv2.THRESH_TOZERO_INV, white_threshold: int = 250, empty_pixels_allowed: int = 0) -> None:
        self.threshold_method = threshold_method
        self.thresh_method = getattr(self, self.threshold_method)
        self.cv2_threshold_method = cv2_threshold_method
        self.white_threshold = white_threshold
        self.empty_pixels_allowed = empty_pixels_allowed

    def measure_of_brightness(self, image: np.ndarray) -> int:
        return self.thresh_method(image)

    def num_pixels_below_threshold(self, image: np.ndarray) -> int:
        gray = BlurDetection.to_gray(image)
        (_, thresholded_image) = cv2.threshold(src=gray, thresh=self.white_threshold, maxval=255, type=self.cv2_threshold_method)
        (h, w) = thresholded_image.shape
        num_non_zero = cv2.countNonZero(thresholded_image)
        num_zero = (h * w) - num_non_zero
        return num_zero

    def percent_pixels_below_threshold(self, image: np.ndarray) -> float:
        gray = BlurDetection.to_gray(image)
        (_, thresholded_image) = cv2.threshold(src=gray, thresh=self.white_threshold, maxval=255, type=self.cv2_threshold_method)
        (h, w) = thresholded_image.shape
        num_non_zero = cv2.countNonZero(thresholded_image)
        num_zero = (h * w) - num_non_zero
        # normalize for image size
        return num_zero / (h * w)

    def too_bright(self, image: np.ndarray) -> bool:
        num_zero = self.num_pixels_below_threshold(image)
        if num_zero > self.empty_pixels_allowed:
            return True
        return False

