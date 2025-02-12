import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import (
    r2_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


# -------------------------------
# 1. Configure GPU
# -------------------------------
def configure_gpu():
    """Configure GPU for optimal usage"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Allow memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPUs found. Running on CPU.")

# Call the configuration function
configure_gpu()

# Replace the device variable with GPU device
device = "/GPU:0"

# -------------------------------
# 2. Create folder for saving diagrams
# -------------------------------
diagram_folder = "diagrams"
os.makedirs(diagram_folder, exist_ok=True)

# -------------------------------
# 3. Data Loading and Preprocessing
# -------------------------------
df = pd.read_csv(
    "kaggle/input/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset/IND_and_Nep_AQI_Dataset.csv"
)
df = shuffle(df)
df = df.sample(frac=1).reset_index(drop=True)

# Optionally split into fragments and use one fragment:
number_of_rows = 3000
sub_dfs = [df[i : i + number_of_rows] for i in range(0, df.shape[0], number_of_rows)]
for idx, sub_df in enumerate(sub_dfs):
    sub_df.to_csv(f"frag3000_{idx}.csv", index=False)

df = pd.read_csv("frag3000_1.csv")


def build_x(path):
    train_img = []
    for i in range(df.shape[0]):
        img = image.load_img(path + df["Filename"][i])
        img = image.img_to_array(img)
        img = tf.keras.applications.vgg16.preprocess_input(img)
        train_img.append(img)
    x = np.array(train_img)
    return x


x_origin = build_x(
    "kaggle/input/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset/All_img/"
)
pm10 = pd.DataFrame(df["PM10"])

# Split into training, validation, and test sets
x_origin_train, x_origin_temp, y_train, y_temp = train_test_split(
    x_origin, pm10, train_size=0.8, shuffle=True
)
x_origin_valid, x_origin_test, y_valid, y_test = train_test_split(
    x_origin_temp, y_temp, test_size=0.5, shuffle=True
)

# -------------------------------
# 4. Build the Model (using VGG16 as feature extractor)
# -------------------------------
with tf.device(device):
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

    opt = Adam(learning_rate=0.0001)
    model.compile(loss="mse", optimizer=opt)

model.summary()

# -------------------------------
# 5. Training Setup
# -------------------------------
weight_path = "vgg16_aqi.best.weights.h5"
callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, verbose=1, mode="auto"),
    ModelCheckpoint(
        weight_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_weights_only=True,
    ),
]


batch_size = 1

with tf.device(device):
    history = model.fit(
        x=x_origin_train,
        y=y_train,
        validation_data=(x_origin_valid, y_valid),
        batch_size=batch_size,
        epochs=150,
        callbacks=callbacks,
    )

# -------------------------------
# 6. Save Training History Plot (Loss Curve)
# -------------------------------
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper left")
loss_plot_path = os.path.join(diagram_folder, "model_loss.png")
plt.savefig(loss_plot_path)
plt.close()
print(f"Loss plot saved to {loss_plot_path}")

# -------------------------------
# 7. Evaluate the Model
# -------------------------------
model.load_weights(weight_path)
loss = model.evaluate(x=x_origin_test, y=y_test, batch_size=16)
print("RMSE is :", loss**0.5)
y_predict = model.predict(x_origin_test)

# -------------------------------
# 8. Post-process Predictions into Categories
# -------------------------------
# Map continuous PM10 predictions into discrete categories
y_predict_pm10 = np.zeros(len(y_predict))
for i in range(len(y_predict)):
    if y_predict[i] <= 54:
        y_predict_pm10[i] = 0
    elif 55 <= y_predict[i] <= 154:
        y_predict_pm10[i] = 1
    elif 155 <= y_predict[i] <= 254:
        y_predict_pm10[i] = 2
    elif 255 <= y_predict[i] <= 354:
        y_predict_pm10[i] = 3
    elif 355 <= y_predict[i] <= 424:
        y_predict_pm10[i] = 4
    elif y_predict[i] > 424:
        y_predict_pm10[i] = 5
    else:
        print("Exception Occurred!")

y_predict_pm10 = y_predict_pm10.astype(int)

# Process ground truth in the same way
y_test = y_test.reset_index(drop=True).to_numpy().tolist()
y_test_pm10 = np.zeros(len(y_test))
for i in range(len(y_test)):
    val = int(y_test[i][0])
    if val <= 54:
        y_test_pm10[i] = 0
    elif 55 <= val <= 154:
        y_test_pm10[i] = 1
    elif 155 <= val <= 254:
        y_test_pm10[i] = 2
    elif 255 <= val <= 354:
        y_test_pm10[i] = 3
    elif 355 <= val <= 424:
        y_test_pm10[i] = 4
    elif val > 424:
        y_test_pm10[i] = 5
    else:
        print("Exception Occurred!")
y_test_pm10 = y_test_pm10.astype(int)

# Calculate metrics
balanced_acc = balanced_accuracy_score(y_test_pm10, y_predict_pm10)
print("Balanced Accuracy:", balanced_acc)

# Manual accuracy calculation
correct = sum(
    1 for i in range(len(y_predict_pm10)) if y_predict_pm10[i] == y_test_pm10[i]
)
acc = correct / len(y_predict_pm10)
print("Accuracy: ", acc, " (", correct, " correct out of", len(y_predict_pm10), ")")

f1 = f1_score(y_test_pm10, y_predict_pm10, average="macro")
print("F1 Score (macro):", f1)

# -------------------------------
# 9. Save Confusion Matrix Plot
# -------------------------------
confusion_mtx = confusion_matrix(y_test_pm10, y_predict_pm10)
plt.figure(figsize=(8, 8))
sns.heatmap(
    confusion_mtx,
    annot=True,
    linewidths=0.01,
    cmap="Greens",
    linecolor="gray",
    fmt=".1f",
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
conf_mat_path = os.path.join(diagram_folder, "confusion_matrix.png")
plt.savefig(conf_mat_path)
plt.close()
print(f"Confusion matrix saved to {conf_mat_path}")

# -------------------------------
# 10. Save Comparison Plot (True vs. Estimation)
# -------------------------------
plt.figure()
plt.plot(y_test, label="True Label")
plt.plot(y_predict, label="Estimation Value")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("True vs. Estimation")
plt.legend()
comparison_plot_path = os.path.join(diagram_folder, "true_vs_estimation.png")
plt.savefig(comparison_plot_path)
plt.close()
print(f"Comparison plot saved to {comparison_plot_path}")
