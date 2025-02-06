# 🌍 Satellite & Aerial Image Analysis

A **Python-based system** for analyzing **satellite imagery** and **aerial photos**, featuring **deforestation detection** and **pollution source identification**.

---

## 🚀 Features

### 🛰️ 1. Satellite Imagery Analysis

-   📊 Detects and quantifies **land use changes** between two time periods
-   🌲 Identifies **deforestation** and **afforestation** patterns
-   🎨 Generates **visualization masks and overlays**
-   📈 Calculates **percentage changes** in forest coverage
-   🛠️ Supports **morphological operations** for noise reduction

### 🏭 2. Pollution Source Detection

-   🔍 Uses **YOLOv5** object detection model
-   🏭 Identifies **pollution sources**, including:
    -   🚗 **Vehicles** (cars, trucks, buses)
    -   🚆 **Transportation** (trains, boats, airplanes)
    -   🏭 **Industrial sources**
-   📍 Generates **impact area visualizations**
-   ✅ Provides **confidence scores** for detections

---

## 🛠 Installation

### 1️⃣ Create a Virtual Environment

```sh
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 2️⃣ Install Dependencies

```sh
pip install -r requirements.txt
```

---

## 📌 Usage

### 🛰️ **Satellite Imagery Analysis**

1. Place your satellite images in the `SatelliteImageryAnalysis/` directory.
2. Run the analysis:
    ```sh
    python SatelliteImageryAnalysis.py
    ```
3. The script will:
    - 📥 Load and register image pairs
    - 🔍 Detect land use changes
    - 🎨 Generate visualization plots
    - 💾 Save results to `output/`

### 🏭 **Air Pollution Analysis**

1. Place your aerial images in the `AnalyzeImages/` directory.
2. Run the detection:
    ```sh
    python AnalyzeImages.py
    ```
3. The script will:
    - 🏭 Detect pollution sources
    - 📍 Create **impact area visualizations**
    - 💾 Save results to `output/`

---

## 📂 Project Structure

```
📁 Project Root
├── 📁 AnalyzeImages/              # Pollution source detection
│   ├── 📁 output/                 # Pollution detection results
│   └── 📜 AnalyzeImages.py        # Pollution analysis script
├── 📁 SatelliteImageryAnalysis/   # Land use change detection
│   ├── 📁 output/                 # Deforestation results
│   └── 📜 SatelliteImageryAnalysis.py  # Analysis script
├── 📁 image/                      # Documentation images
├── 📜 requirements.txt            # Project dependencies
└── 📜 yolov5s.pt                  # Pretrained YOLOv5 model
```

---

## 📦 Dependencies

-   🖼 **OpenCV** – Image processing
-   🔢 **NumPy** – Numerical operations
-   🔥 **PyTorch** – Deep learning
-   📊 **Matplotlib** – Data visualization
-   🔍 **YOLOv5** – Object detection

See **`requirements.txt`** for the full list.

---

## 📖 Documentation

-   📄 **[Satellite Imagery Analysis Documentation](./SatelliteImageryAnalysis.md)**
-   📄 **[Pollution Detection Documentation](./AnalyzeImages.md)**

---

## ⚡ Performance Considerations

-   🖥️ **RAM Usage:** High-resolution satellite images require substantial memory.
-   ⚡ **GPU Recommended:** Faster inference for YOLOv5.
-   🖼 **High-Resolution Outputs:** Processed images are saved at **300 DPI**.

---

## ⚠️ Known Limitations

-   📍 **Pre-Aligned Images Required:** Satellite images must be pre-registered.
-   🎚 **Fixed Thresholds:** Change detection relies on preset values.
-   🌈 **Limited Spectrum Analysis:** Only visible spectrum is currently supported.

---

## 🔮 Future Improvements

-   🛰 **Automated Image Registration** (Feature Matching, Deep Learning)
-   🌍 **Multi-Spectral Imagery Support** (e.g., NDVI for vegetation health)
-   🤖 **Machine Learning-Based Classification** (Beyond threshold-based detection)
-   📦 **Batch Processing Capabilities** (Automated large-scale analysis)
-   📊 **Progress Tracking & Reporting Features**

---

💡 **Author:** _Vivek Sharma_
