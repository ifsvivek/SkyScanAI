import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from haze_features import HazeFeatureExtractor


class AQIEstimator(nn.Module):
    def __init__(self):
        super(AQIEstimator, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.feature_size = 128 * (256 // (2**3)) * (256 // (2**3))

        # Dense layers with batch normalization
        self.fc1 = nn.Linear(self.feature_size + 6, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.3)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, haze_features):
        # CNN layers with batch normalization
        x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
        x = F.relu(self.bn2(F.max_pool2d(self.conv2(x), 2)))
        x = F.relu(self.bn3(F.max_pool2d(self.conv3(x), 2)))

        x = x.view(-1, self.feature_size)
        x = torch.cat((x, haze_features), dim=1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        # Constrain output to valid AQI range (0-500)
        x = torch.sigmoid(x) * 500

        return x


def get_aqi_color(aqi_value):
    """Get color coding for AQI value."""
    if aqi_value <= 50:
        return (0, 255, 0)  # Green - Good
    elif aqi_value <= 100:
        return (0, 255, 255)  # Yellow - Moderate
    elif aqi_value <= 150:
        return (0, 165, 255)  # Orange - Unhealthy for Sensitive Groups
    elif aqi_value <= 200:
        return (0, 0, 255)  # Red - Unhealthy
    elif aqi_value <= 300:
        return (128, 0, 128)  # Purple - Very Unhealthy
    else:
        return (0, 0, 128)  # Maroon - Hazardous


def add_aqi_overlay(img, aqi_value):
    """Add AQI index overlay to image."""
    # Make a copy of the image
    output = img.copy()

    # Get color based on AQI value
    color = get_aqi_color(aqi_value)

    # Add background rectangle for better visibility
    cv2.rectangle(output, (10, 10), (200, 60), (255, 255, 255), -1)
    cv2.rectangle(output, (10, 10), (200, 60), color, 2)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output, f"AQI: {aqi_value:.1f}", (20, 45), font, 1, color, 2)

    return output


class AQIPredictor:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AQIEstimator().to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Updated feature normalization parameters
        self.feature_means = {
            "dark_channel_mean": 0.3,
            "transmission_mean": 0.7,
            "atmospheric_light": 0.8,
            "contrast": 35.0,
            "saturation": 100.0,
            "haze_density": 0.3,
        }
        self.feature_stds = {
            "dark_channel_mean": 0.2,
            "transmission_mean": 0.2,
            "atmospheric_light": 0.15,
            "contrast": 15.0,
            "saturation": 50.0,
            "haze_density": 0.2,
        }

    def normalize_features(self, features):
        """Normalize feature values."""
        normalized = {}
        for key in features:
            normalized[key] = (
                features[key] - self.feature_means[key]
            ) / self.feature_stds[key]
        return normalized

    def preprocess_image(self, img):
        """Preprocess image for model input."""
        # Resize to 256x256
        img = cv2.resize(img, (256, 256))
        # Normalize
        img = img.astype(np.float32) / 255.0
        # To tensor
        img = torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0)
        return img

    def calculate_base_aqi(self, features):
        """Calculate base AQI from haze features using clear air criteria."""
        # Clear air indicators
        is_clear = (
            features["transmission_mean"] > 0.8
            and features["haze_density"] < 0.2
            and features["contrast"] > 35.0
        )

        if is_clear:
            return 25  # Good AQI for clear air

        # Calculate weighted AQI based on features
        haze_score = (
            0.3 * (1 - features["transmission_mean"])
            + 0.3 * features["haze_density"]
            + 0.2 * (features["dark_channel_mean"])
            + 0.2 * (1 - (features["contrast"] / 100))
        ) * 500

        return min(500, max(0, haze_score))

    def predict_aqi(self, img, haze_features):
        """Predict AQI from image and haze features."""
        with torch.no_grad():
            # Calculate base AQI from features
            base_aqi = self.calculate_base_aqi(haze_features)

            # Use model for fine-tuning
            img_tensor = self.preprocess_image(img).to(self.device)
            normalized_features = self.normalize_features(haze_features)
            feature_values = [
                normalized_features[key]
                for key in [
                    "dark_channel_mean",
                    "transmission_mean",
                    "atmospheric_light",
                    "contrast",
                    "saturation",
                    "haze_density",
                ]
            ]

            feature_tensor = torch.FloatTensor([feature_values]).to(self.device)
            model_prediction = self.model(img_tensor, feature_tensor).item()

            # Blend base AQI with model prediction
            final_aqi = 0.7 * base_aqi + 0.3 * model_prediction

            # Apply clear air rules
            if (
                haze_features["transmission_mean"] > 0.85
                and haze_features["haze_density"] < 0.15
                and haze_features["contrast"] > 40.0
            ):
                final_aqi = min(final_aqi, 50)  # Cap at 50 for very clear air

            return round(min(500, max(0, final_aqi)), 2)

    def process_and_save(self, img, output_path):
        """Process image and save result with AQI overlay."""
        # Extract features and predict AQI
        extractor = HazeFeatureExtractor()  # Create instance here
        features = extractor.extract_features(img)
        aqi = self.predict_aqi(img, features)

        # Add AQI overlay
        output_img = add_aqi_overlay(img, aqi)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the image
        cv2.imwrite(output_path, output_img)

        return aqi, features


def main():

    # Use absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "image.png")
    output_dir = os.path.join(os.path.dirname(current_dir), "ImageAQI/output")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "result_" + os.path.basename(img_path))

    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return

    try:
        # Process image and save result
        predictor = AQIPredictor()
        aqi, features = predictor.process_and_save(img, output_path)

        print("\nExtracted Features:")
        for key, value in features.items():
            print(f"{key}: {value}")
        print(f"\nEstimated AQI: {aqi}")
        print(f"Output saved to: {output_path}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")


if __name__ == "__main__":
    main()
