import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")


def create_model():
    """Recreate the model architecture"""
    pre_trained_model = tf.keras.applications.VGG16(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    for layer in pre_trained_model.layers:
        layer.trainable = False

    x1 = Flatten()(pre_trained_model.output)
    fc1 = Dense(512, activation="relu")(x1)
    fc2 = Dense(512, activation="relu")(fc1)
    output = Dense(1, activation="linear")(fc2)
    model = Model(pre_trained_model.input, output)

    return model


def preprocess_image(img_path):
    """Load and preprocess a single image"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


def get_aqi_category(pm10_value):
    """Convert PM10 value to AQI category"""
    if pm10_value <= 54:
        return "Good (0)"
    elif pm10_value <= 154:
        return "Moderate (1)"
    elif pm10_value <= 254:
        return "Unhealthy for Sensitive Groups (2)"
    elif pm10_value <= 354:
        return "Unhealthy (3)"
    elif pm10_value <= 424:
        return "Very Unhealthy (4)"
    else:
        return "Hazardous (5)"


def evaluate_image(model, image_path):
    """Evaluate a single image and return its AQI prediction"""
    try:
        processed_image = preprocess_image(image_path)
        with tf.device("/CPU:0"):  # Force CPU execution
            prediction = model.predict(processed_image, verbose=0)[0][
                0
            ]  # Reduce verbosity
        category = get_aqi_category(prediction)
        return prediction, category
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")


def main():
    # Load the model and weights
    try:
        with tf.device("/CPU:0"):  # Force CPU execution
            model = create_model()
            weights_path = "vgg16_aqi.best.weights.h5"

            if not os.path.exists(weights_path):
                print(f"Error: Model weights file not found at {weights_path}")
                return

            model.load_weights(weights_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Directory containing images to evaluate
    image_dir = "ImageAQI"

    if not os.path.exists(image_dir):
        print(f"Creating test images directory: {image_dir}")
        os.makedirs(image_dir)
        print(f"Please place your test images in the {image_dir} directory")
        return

    # Process all images in the directory
    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_dir, image_file)
            try:
                pm10_value, category = evaluate_image(model, image_path)
                print(f"\nImage: {image_file}")
                print(f"Predicted PM10 Value: {pm10_value:.2f}")
                print(f"AQI Category: {category}")
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")


if __name__ == "__main__":
    main()
