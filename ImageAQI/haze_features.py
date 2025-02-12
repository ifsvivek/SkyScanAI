import cv2
import numpy as np
from scipy.ndimage import median_filter


class HazeFeatureExtractor:
    def __init__(self):
        self.dark_channel_window = 15
        self.transmission_window = 15
        self.omega = 0.95  # Transmission estimate parameter

    def get_dark_channel(self, img):
        """Extract dark channel from image."""
        b, g, r = cv2.split(img)
        dc = cv2.min(cv2.min(r, g), b)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.dark_channel_window, self.dark_channel_window)
        )
        dark_channel = cv2.erode(dc, kernel)
        return dark_channel

    def estimate_atmospheric_light(self, img, dark_channel):
        """Estimate atmospheric light in hazy image."""
        h, w = dark_channel.shape
        flat_img = img.reshape(h * w, 3)
        flat_dark = dark_channel.ravel()

        # Pick top 0.1% brightest pixels in the dark channel
        searchidx = (-flat_dark).argsort()[: int(h * w * 0.001)]
        atmospheric_light = np.max(flat_img.take(searchidx, axis=0), axis=0)

        return atmospheric_light

    def estimate_transmission(self, img, atmospheric_light):
        """Estimate transmission map."""
        normalized = img / atmospheric_light
        transmission = 1 - self.omega * self.get_dark_channel(normalized)
        return median_filter(transmission, self.transmission_window)

    def extract_features(self, img):
        """Extract haze-relevant features from image."""
        # Convert to float and normalize
        img_norm = img.astype(np.float32) / 255.0

        # Get dark channel
        dark_channel = self.get_dark_channel(img_norm)

        # Estimate atmospheric light
        atmospheric_light = self.estimate_atmospheric_light(img_norm, dark_channel)
        atmospheric_light_mean = float(np.mean(atmospheric_light))

        # Estimate transmission map
        transmission = self.estimate_transmission(img_norm, atmospheric_light)

        # Calculate average contrast
        contrast = float(np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

        # Calculate average saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = float(np.mean(hsv[:, :, 1]))

        # Calculate haze density estimation
        haze_density = float(1 - np.mean(transmission))

        return {
            "dark_channel_mean": float(np.mean(dark_channel)),
            "transmission_mean": float(np.mean(transmission)),
            "atmospheric_light": atmospheric_light_mean,
            "contrast": contrast,
            "saturation": saturation,
            "haze_density": haze_density,
        }


def main():
    # Test the feature extractor
    import os

    # Use absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "image.png")

    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return

    extractor = HazeFeatureExtractor()
    features = extractor.extract_features(img)

    print("Extracted Features:")
    for key, value in features.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
