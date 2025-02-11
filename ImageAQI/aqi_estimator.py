import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class AQIEstimator(nn.Module):
    def __init__(self):
        super(AQIEstimator, self).__init__()

        # CNN for feature extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Calculate the size after convolutions and pooling
        self.feature_size = 128 * (256 // (2**3)) * (256 // (2**3))  # 128 * 32 * 32

        # Dense layers for AQI prediction
        self.fc1 = nn.Linear(self.feature_size + 6, 256)  # 6 is the number of haze features
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, haze_features):
        # Process image through CNN
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))

        # Flatten CNN output
        x = x.view(-1, self.feature_size)

        # Concatenate with haze features
        x = torch.cat((x, haze_features), dim=1)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class AQIPredictor:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AQIEstimator().to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def preprocess_image(self, img):
        """Preprocess image for model input."""
        # Resize to 256x256
        img = cv2.resize(img, (256, 256))
        # Normalize
        img = img.astype(np.float32) / 255.0
        # To tensor
        img = torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0)
        return img

    def predict_aqi(self, img, haze_features):
        """Predict AQI from image and haze features."""
        with torch.no_grad():
            img_tensor = self.preprocess_image(img).to(self.device)
            
            # Extract features in the correct order
            feature_values = [
                haze_features['dark_channel_mean'],
                haze_features['transmission_mean'],
                haze_features['atmospheric_light'],
                haze_features['contrast'],
                haze_features['saturation'],
                haze_features['haze_density']
            ]
            
            # Convert to tensor properly
            feature_tensor = torch.FloatTensor([feature_values]).to(self.device)
            
            prediction = self.model(img_tensor, feature_tensor)
            return prediction.item()


def main():
    # Test the AQI predictor
    import os
    from haze_features import HazeFeatureExtractor

    # Use absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "i2.png")
    
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return
        
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return

    # Extract haze features
    try:
        extractor = HazeFeatureExtractor()
        features = extractor.extract_features(img)

        # Predict AQI
        predictor = AQIPredictor()
        aqi = predictor.predict_aqi(img, features)

        print(f"Estimated AQI: {aqi}")
    except Exception as e:
        print(f"Error during processing: {str(e)}")


if __name__ == "__main__":
    main()
