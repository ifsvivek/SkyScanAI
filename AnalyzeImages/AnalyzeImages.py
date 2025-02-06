import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_image(image_path):
    """
    Load an image from disk and convert it from BGR to RGB.
    """
    # Read image with OpenCV (BGR format)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    # Convert BGR to RGB for correct color representation in matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def create_pollution_mask(img, detections):
    """
    Create a mask highlighting potential pollution areas.
    """
    mask = np.zeros_like(img)
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        # Fill the detection area with red color with 50% transparency
        mask[y1:y2, x1:x2] = [255, 0, 0]
    return mask


def display_detections(img, detections, class_names):
    """
    Draw bounding boxes and pollution overlay on the image.
    """
    img_draw = img.copy()

    # Create and apply pollution mask
    mask = create_pollution_mask(img, detections)
    img_draw = cv2.addWeighted(img_draw, 1, mask, 0.3, 0)

    # Draw bounding boxes
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{class_names[int(cls)]} {conf:.2f}"

        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=2)
        cv2.putText(
            img_draw,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            thickness=2,
        )

    plt.figure(figsize=(20, 12), dpi=300)  # Increased figure size and DPI
    plt.imshow(img_draw)
    plt.axis("off")
    plt.title("Detected Pollution Sources with Impact Areas")
    plt.savefig(
        "AnalyzeImages/output/output.png", dpi=300, bbox_inches="tight", pad_inches=0
    )


def filter_pollution_sources(detections_df):
    """
    Filter detections to only include air pollution sources.
    """
    # Define classes that are typically pollution sources
    pollution_sources = [
        "car",
        "truck",
        "bus",
        "motorcycle",
        "train",
        "boat",
        "airplane",
    ]

    # Filter the DataFrame to only include pollution sources
    filtered_df = detections_df[detections_df["name"].isin(pollution_sources)]
    return filtered_df


def main():
    # Replace with your local image path
    image_path = "AnalyzeImages/2.jpg"

    # Load the image
    img = load_image(image_path)

    # Load the pretrained YOLOv5s model from the Ultralytics repository.
    # Source: https://github.com/ultralytics/yolov5
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    # Run inference on the image.
    results = model(img)

    # Print the inference results to the console.
    results.print()

    # Convert results to a pandas DataFrame.
    detections_df = results.pandas().xyxy[0]
    detections_df = filter_pollution_sources(detections_df)
    print("Detected pollution sources:")
    print(detections_df)

    # Extract filtered detections as a list
    detections = []
    for _, row in detections_df.iterrows():
        detections.append(
            [
                row["xmin"],
                row["ymin"],
                row["xmax"],
                row["ymax"],
                row["confidence"],
                row["class"],
            ]
        )

    # Get the mapping of class indices to names
    class_names = results.names

    # Display the image with the detected bounding boxes.
    display_detections(img, detections, class_names)


if __name__ == "__main__":
    main()
