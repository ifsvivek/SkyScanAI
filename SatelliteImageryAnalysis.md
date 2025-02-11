# **Satellite Imagery Analysis System**

## **1. Core Concept**

System for analyzing changes between two satellite images to detect deforestation and afforestation.

## **2. Technical Implementation**

### **2.1 Core Functions**

```python
def load_image(image_path):
    # Loads and converts BGR to RGB
    return img_rgb

def register_images(img1, img2):
    # Validates matching dimensions
    return img1, img2

def change_detection(img1, img2, threshold_deforestation=-20,
                    threshold_afforestation=20, kernel_size=5):
    # Detects changes using HSV color space
    return deforestation_mask, afforestation_mask

def analyze_changes(deforestation_mask, afforestation_mask):
    # Calculates change percentages
    return percent_deforestation, percent_afforestation
```

### **2.2 Change Detection Parameters**

-   `threshold_deforestation`: -20 (brightness decrease)
-   `threshold_afforestation`: 20 (brightness increase)
-   `kernel_size`: 5x5 (morphological operations)

### **2.3 Visualization Components**

Six-panel output showing:

1. Original image (Time 1)
2. Original image (Time 2)
3. Change overlay
4. Deforestation mask
5. Afforestation mask
6. Contour annotation

## **3. Usage Example**

```python
# Load images
img1 = load_image("2009.png")
img2 = load_image("2019.png")

# Detect changes
deforestation, afforestation = change_detection(img1, img2)

# Get statistics
defo_pct, affo_pct = analyze_changes(deforestation, afforestation)
print(f"Deforestation: {defo_pct:.2f}%")
print(f"Afforestation: {affo_pct:.2f}%")
```

## **4. Technical Requirements**

-   Python 3.7+
-   OpenCV
-   NumPy
-   Matplotlib

## **5. Limitations**

-   Requires pre-aligned images
-   Uses fixed thresholds
-   Limited to RGB imagery
