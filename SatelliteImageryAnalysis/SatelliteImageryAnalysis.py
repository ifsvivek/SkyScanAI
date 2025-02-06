import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_image(image_path):
    """
    Load an image from disk using OpenCV and convert from BGR to RGB.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: RGB image.
    """
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def register_images(img1, img2):
    """
    Placeholder for image registration. Assumes that the images are already aligned.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Registered images.
    """
    if img1.shape != img2.shape:
        raise ValueError(
            "Input images must have the same dimensions for proper registration."
        )
    return img1, img2


def change_detection(
    img1, img2, threshold_deforestation=-20, threshold_afforestation=20, kernel_size=5
):
    """
    Detect changes between two images by analyzing the Value (V) channel in HSV color space.
    A decrease in brightness indicates potential deforestation, while an increase indicates afforestation.

    Args:
        img1 (np.ndarray): First image (earlier date).
        img2 (np.ndarray): Second image (later date).
        threshold_deforestation (int): Threshold for decrease in V channel to indicate deforestation.
        threshold_afforestation (int): Threshold for increase in V channel to indicate afforestation.
        kernel_size (int): Size of the kernel for morphological operations.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Cleaned binary masks for deforestation and afforestation.
    """
    # Convert images to HSV color space
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)

    # Extract the Value channel as a proxy for vegetation intensity
    value1 = hsv1[:, :, 2]
    value2 = hsv2[:, :, 2]

    # Compute the difference in the Value channel (later - earlier)
    diff_value = value2.astype(np.int16) - value1.astype(np.int16)

    # Initialize binary masks for deforestation and afforestation
    deforestation_mask = np.zeros_like(diff_value, dtype=np.uint8)
    afforestation_mask = np.zeros_like(diff_value, dtype=np.uint8)

    # Mark pixels where the decrease/increase in brightness exceeds thresholds
    deforestation_mask[diff_value < threshold_deforestation] = 255
    afforestation_mask[diff_value > threshold_afforestation] = 255

    # Create a kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Clean up the masks using morphological opening and closing
    deforestation_mask_clean = cv2.morphologyEx(
        deforestation_mask, cv2.MORPH_OPEN, kernel
    )
    deforestation_mask_clean = cv2.morphologyEx(
        deforestation_mask_clean, cv2.MORPH_CLOSE, kernel
    )

    afforestation_mask_clean = cv2.morphologyEx(
        afforestation_mask, cv2.MORPH_OPEN, kernel
    )
    afforestation_mask_clean = cv2.morphologyEx(
        afforestation_mask_clean, cv2.MORPH_CLOSE, kernel
    )

    return deforestation_mask_clean, afforestation_mask_clean


def analyze_changes(deforestation_mask, afforestation_mask):
    """
    Calculate the percentage of the area that has changed for both deforestation and afforestation.

    Args:
        deforestation_mask (np.ndarray): Binary mask for deforestation.
        afforestation_mask (np.ndarray): Binary mask for afforestation.

    Returns:
        Tuple[float, float]: Percentage of deforestation and afforestation.
    """

    def compute_percentage(mask):
        total_area = mask.shape[0] * mask.shape[1]
        changed_area = np.count_nonzero(mask)
        return (changed_area / total_area) * 100

    percent_deforestation = compute_percentage(deforestation_mask)
    percent_afforestation = compute_percentage(afforestation_mask)
    return percent_deforestation, percent_afforestation


def draw_contours(image, deforestation_mask, afforestation_mask):
    """
    Draw contours on the image from the provided binary masks.
    Red contours indicate deforestation and green contours indicate afforestation.

    Args:
        image (np.ndarray): Original RGB image.
        deforestation_mask (np.ndarray): Binary mask for deforestation.
        afforestation_mask (np.ndarray): Binary mask for afforestation.

    Returns:
        np.ndarray: Annotated image with drawn contours.
    """
    annotated_img = image.copy()
    # Convert to BGR for drawing (OpenCV uses BGR)
    annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

    # Find contours for deforestation (red)
    contours_deforestation, _ = cv2.findContours(
        deforestation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(annotated_img_bgr, contours_deforestation, -1, (0, 0, 255), 2)

    # Find contours for afforestation (green)
    contours_afforestation, _ = cv2.findContours(
        afforestation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(annotated_img_bgr, contours_afforestation, -1, (0, 255, 0), 2)

    # Convert back to RGB for display
    annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
    return annotated_img_rgb


def overlay_mask(image, deforestation_mask, afforestation_mask, alpha=0.5):
    """
    Overlay the deforestation and afforestation masks on the image.
    Red overlay indicates deforestation and green overlay indicates afforestation.

    Args:
        image (np.ndarray): Original RGB image.
        deforestation_mask (np.ndarray): Binary mask for deforestation.
        afforestation_mask (np.ndarray): Binary mask for afforestation.
        alpha (float): Blending factor.

    Returns:
        np.ndarray: Image with overlays.
    """
    overlay = image.copy()

    # Create color masks for deforestation (red) and afforestation (green)
    red_mask = np.zeros_like(image)
    red_mask[:, :, 0] = 255  # Red channel

    green_mask = np.zeros_like(image)
    green_mask[:, :, 1] = 255  # Green channel

    # Convert masks to boolean arrays
    defo_bool = deforestation_mask.astype(bool)
    affo_bool = afforestation_mask.astype(bool)

    # Blend the red mask on areas of deforestation
    overlay[defo_bool] = cv2.addWeighted(
        overlay[defo_bool], 1 - alpha, red_mask[defo_bool], alpha, 0
    )

    # Blend the green mask on areas of afforestation
    overlay[affo_bool] = cv2.addWeighted(
        overlay[affo_bool], 1 - alpha, green_mask[affo_bool], alpha, 0
    )

    return overlay


def plot_results(
    img1,
    img2,
    overlay_img,
    deforestation_mask,
    afforestation_mask,
    annotated_img,
    percent_deforestation,
    percent_afforestation,
    save_path=None,
):
    """
    Plot the original images, overlay, masks, and annotated image with percentage changes.
    Optionally save the plot to disk.

    Args:
        img1 (np.ndarray): Image from time 1.
        img2 (np.ndarray): Image from time 2.
        overlay_img (np.ndarray): Image with overlay masks.
        deforestation_mask (np.ndarray): Binary deforestation mask.
        afforestation_mask (np.ndarray): Binary afforestation mask.
        annotated_img (np.ndarray): Image annotated with contours.
        percent_deforestation (float): Percentage change for deforestation.
        percent_afforestation (float): Percentage change for afforestation.
        save_path (str, optional): Path to save the resulting plot. Defaults to None.
    """
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))

    axs[0, 0].imshow(img1)
    axs[0, 0].set_title("Satellite Image - Time 1")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(img2)
    axs[0, 1].set_title("Satellite Image - Time 2")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(overlay_img)
    axs[0, 2].set_title("Overlay (Red=Deforestation, Green=Afforestation)")
    axs[0, 2].axis("off")

    axs[1, 0].imshow(deforestation_mask, cmap="gray")
    axs[1, 0].set_title("Deforestation Mask")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(afforestation_mask, cmap="gray")
    axs[1, 1].set_title("Afforestation Mask")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(annotated_img)
    axs[1, 2].set_title("Annotated Changes (Contours)")
    axs[1, 2].axis("off")

    # Add percentage change text
    plt.suptitle(
        f"Deforestation: {percent_deforestation:.2f}% | Afforestation: {percent_afforestation:.2f}%",
        fontsize=20,
        y=0.95,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def main():
    # ------------------------ User Settings ------------------------
    image1_path = "SatelliteImageryAnalysis/2009.png"  # Image from an earlier date
    image2_path = "SatelliteImageryAnalysis/2019.png"  # Image from a later date
    output_plot_path = "SatelliteImageryAnalysis/output/NEW_AD(2009-2019).png"
    # -----------------------------------------------------------------

    # Load the images
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

    # Register images (assumes they are already aligned)
    img1_aligned, img2_aligned = register_images(img1, img2)

    # Perform change detection (detect both deforestation and afforestation)
    deforestation_mask, afforestation_mask = change_detection(
        img1_aligned,
        img2_aligned,
        threshold_deforestation=-20,
        threshold_afforestation=20,
        kernel_size=5,
    )

    # Calculate the percentage changes
    percent_deforestation, percent_afforestation = analyze_changes(
        deforestation_mask, afforestation_mask
    )
    print(f"Deforestation: {percent_deforestation:.2f}% of the area has changed.")
    print(f"Afforestation: {percent_afforestation:.2f}% of the area has changed.")

    # Generate visualization overlays and annotations
    overlay_img = overlay_mask(
        img2_aligned, deforestation_mask, afforestation_mask, alpha=0.5
    )
    annotated_img = draw_contours(img2_aligned, deforestation_mask, afforestation_mask)

    # Plot and save the results
    plot_results(
        img1_aligned,
        img2_aligned,
        overlay_img,
        deforestation_mask,
        afforestation_mask,
        annotated_img,
        percent_deforestation,
        percent_afforestation,
        save_path=output_plot_path,
    )


if __name__ == "__main__":
    main()
