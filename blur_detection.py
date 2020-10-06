# blur_detection.py

import numpy as np
import cv2


# note: threshold goes 0 to 100. empirically about 14 works...
class BlurDetection:
    def __init__(self, threshold: int = 14, blur_detection_method: str = "singular_value_decomposition"):      # variance_of_laplacian
        self.threshold = threshold
        self.blur_detection_method = blur_detection_method
        self.detection_method = getattr(self, self.blur_detection_method)

    def measure_of_focus(self, image: np.ndarray) -> float:
        # noinspection PyTypeChecker
        gray: np.ndarray = self.to_gray(image)
        return self.detection_method(gray)

    def is_blurry(self, image: np.ndarray) -> bool:
        return measure_of_focus(image) < self.threshold

    # Available methods to estimate blurriness
    @staticmethod
    def variance_of_laplacian(image: np.ndarray):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()
        # TODO: where do we use the threshold??

    @staticmethod
    def to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim < 2 or image.ndim > 3:
            raise TypeError(f"image provided to to_gray has invalid dimensions: {image.shape}")
        if image.ndim == 2:
            return image
        if image.ndim == 3:
            w, h, ch = image.shape
            if ch == 1:
                return image
            if ch == 2:
                raise TypeError(f"image provided to to_gray has invalid dimensions: {image.shape}")
            if ch == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if ch == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        raise TypeError(f"image provided to to_gray has invalid dimensions: {image.shape}")

    # Singular Value Decomposition Method
    # https://webcache.googleusercontent.com/search?q=cache:eW366WVfDV4J:https://www.akshatvasistha.com/2019/12/real-time-blur-detection-methods-opencv.html+&cd=12&hl=en&ct=clnk&gl=us
    @staticmethod
    def singular_value_decomposition(image: np.ndarray):
        u, s, v = np.linalg.svd(image)
        # TODO: how does the threshold affect the statistics?
        svd_threshold = 5
        top_sv = np.sum(s[0:svd_threshold])
        total_sv = np.sum(s)
        return top_sv / total_sv
