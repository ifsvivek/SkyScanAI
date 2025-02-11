# **Haze Feature Extraction System**

## **1. Feature Components**

### **1.1 Dark Channel Prior**

-   Represents the minimum intensity in a local region
-   Helps identify haze-opaque regions
-   Lower values indicate clearer atmosphere

### **1.2 Transmission Map**

-   Estimates the portion of light that reaches the camera
-   Values closer to 1 indicate less haze
-   Used to quantify haze density

### **1.3 Atmospheric Light**

-   Estimates the intensity of atmospheric light
-   Higher values indicate more scattered light
-   Key indicator of haze presence

### **1.4 Image Statistics**

-   Contrast: Measures intensity variation
-   Saturation: Color intensity measurement
-   Haze Density: Combined metric of opacity

## **2. Usage Example**

```python
from haze_features import HazeFeatureExtractor

# Initialize extractor
extractor = HazeFeatureExtractor()

# Extract features from image
features = extractor.extract_features(image)

# Access individual features
dark_channel = features['dark_channel_mean']
transmission = features['transmission_mean']
atmospheric = features['atmospheric_light']
contrast = features['contrast']
saturation = features['saturation']
haze_density = features['haze_density']
```

## **3. Feature Ranges**

| Feature           | Range    | Interpretation   |
| ----------------- | -------- | ---------------- |
| Dark Channel Mean | [0, 1]   | Lower = Clearer  |
| Transmission Mean | [0, 1]   | Higher = Clearer |
| Atmospheric Light | [0, 255] | Higher = Hazier  |
| Contrast          | [0, 1]   | Higher = Clearer |
| Saturation        | [0, 1]   | Higher = Clearer |
| Haze Density      | [0, 1]   | Higher = Hazier  |

