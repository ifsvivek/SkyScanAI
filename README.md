# SkyScanAI

## Overview

SkyScanAI is an advanced environmental monitoring platform that leverages satellite and aerial imagery to assess changes in land use, monitor deforestation/afforestation trends, and predict air quality levels. The system integrates computer vision, deep learning, and image processing techniques to extract meaningful insights from large-scale remote sensing data.

## Core Concept

The primary goal of SkyScanAI is to support environmental analysis by:

-   Detecting and quantifying changes in vegetation cover.
-   Estimating air quality indices (AQI) using advanced haze feature extraction.
-   Identifying potential pollution sources using object detection.

Each of these tasks is achieved through specialized modules that preprocess imagery, extract features, run analytical models, and provide visualizations and quantitative assessments.

## Detailed Architecture

### 1. Satellite Imagery Analysis

-   Compares temporal image pairs to locate deforestation and afforestation.
-   Utilizes image registration and change detection algorithms based on HSV color space.
-   Provides tools for overlay creation, contour drawing, and percentage change calculations.

### 2. Air Quality Analysis

-   Predicts AQI using a deep convolutional neural network (CNN) enhanced with haze features.
-   Implements haze feature extraction modules to capture image-specific indicators such as dark channel, transmission map, and atmospheric light.
-   Combines CNN-derived image features with extracted haze metrics to generate a robust air quality estimate.

### 3. Pollution Source Detection

-   Applies YOLOv5 object detection to identify and localize common air pollution source categories.
-   Provides visual overlays with bounding boxes and transparency masks to illustrate detected sources.
-   Filters detections to focus on entities typically associated with environmental pollution.

## Modules

-   **SatelliteImageryAnalysis**: Contains scripts for processing and analyzing satellite images, including change detection and visualization.
-   **ImageAQI**: Hosts the AQI prediction model, integrating CNN-based feature extraction with haze feature analysis.
-   **AnalyzeImages**: Implements pollution detection using a pre-trained YOLOv5 model to pinpoint potential pollution sources.

## Technical Specifications

-   **Programming Language**: Python 3.7+
-   **Core Libraries**: OpenCV, NumPy, Matplotlib, PyTorch
-   **Performance**:
    -   Deforestation detection accuracy around 85%.
    -   AQI estimation with a typical RMSE of ±15 AQI points.
    -   High-resolution image support and real-time processing capabilities.
-   **System Limitations**:
    -   Requires pre-aligned satellite images.
    -   Operates primarily on RGB imagery.
    -   Fixed thresholds in some processing steps may necessitate calibration for different environments.

## Model Performance

-   **Satellite Analysis**: Accuracy ~85%, processing time 2-3 seconds per image pair.
-   **AQI Estimation**: RMSE of approximately ±15 AQI points, suitable for real-time monitoring on diverse datasets.
-   **Pollution Detection**: High precision on common sources with enhanced visualization output at 300 DPI.
