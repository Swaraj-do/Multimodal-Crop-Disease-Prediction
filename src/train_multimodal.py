import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

CSV_PATH = "data/multimodal_labels.csv"

IMG_SIZE = (224, 224)
BATCH = 16

# Train in 2 stages
EPOCHS_HEAD = 10        # with early stopping
EPOCHS_FINE = 5         # with early stopping

ENV_COLS = ["temperature", "humidity", "rainfall", "wind_speed", "season_code"]


def load_image(path: tf.Tensor) -> tf.Tensor:
    """
    Safe image loader for TF graphs (Windows-friendly).
    Avoids tf.image.decode_image() which can produce unknown shape tensors.
    Supports JPG/JPEG/PNG.
    """
    img_bytes = tf.io.read_file(path)
    lower = tf.strings.lower(path)

    def decode_png():
        return tf.image.decode_png(img_bytes, channels=3)

    def decode_jpeg():
        return tf.image.decode_jpeg(img_bytes, channels=3)

    # If file ends with .png -> decode_png else decode_jpeg
    img = tf.cond(
        tf.strings.regex_full_match(lower, r".*\.png"),
        true_fn=decode_png,
        false_fn=decode_jpeg
    )

    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_dataset(df: pd.DataFrame, num_classes: int, training: bool = True) -> tf.data.Dataset:
    paths = df["image_path"].values
    env = df[ENV_COLS].values.astype("float32")
    labels = tf.keras.utils.to_categorical(df["label"].values, num_classes)

    ds = tf.data.Dataset.from_tensor_slices((paths, env, labels))

    def process(p, e, y):
        img = load_image(p)
        if training:
            img = tf.image.random_flip_left_right(img)
        return {"img": img, "env": e}, y

    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2048)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(num_classes: int):
    # Image branch
    img_input = tf.keras.Input(shape=(224, 224, 3), name="img")
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=img_input
    )
    base.trainable = False

    x_img = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x_img = tf.keras.layers.Dense(256, activation="relu")(x_img)
    x_img = tf.keras.layers.Dropout(0.3)(x_img)

    # Env branch
    env_input = tf.keras.Input(shape=(len(ENV_COLS),), name="env")
    x_env = tf.keras.layers.Dense(64, activation="relu")(env_input)
    x_env = tf.keras.layers.Dense(32, activation="relu")(x_env)

    # Fusion
    x = tf.keras.layers.Concatenate()([x_img, x_env])
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs={"img": img_input, "env": env_input}, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base


def save_history_plot(histories, out_path: str, title: str):
    """
    histories: list of tf.keras.callbacks.History (head + fine-tune)
    """
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []

    for h in histories:
        train_acc += h.history.get("accuracy", [])
        val_acc += h.history.get("val_accuracy", [])
        train_loss += h.history.get("loss", [])
        val_loss += h.history.get("val_loss", [])

    plt.figure()
    plt.plot(train_acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_acc.png"))
    plt.close()

    plt.figure()
    plt.plot(train_loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_loss.png"))
    plt.close()


def main():
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    num_classes = df["label"].nunique()

    train_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, random_state=42, stratify=train_df["label"]
    )

    train_ds = make_dataset(train_df, num_classes, training=True)
    val_ds = make_dataset(val_df, num_classes, training=False)
    test_ds = make_dataset(test_df, num_classes, training=False)

    model, base = build_model(num_classes)

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    print("\n=== Stage 1: Train Multimodal Head (base frozen) ===\n")
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=[early]
    )

    print("\n=== Stage 2: Fine-tune last 30 layers of MobileNetV2 ===\n")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        callbacks=[early]
    )

    # Evaluate
    y_true, y_pred = [], []
    for x, y in test_ds:
        p = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(p, axis=1))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues")
    plt.title("Multimodal Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("results/multimodal_confusion_matrix.png")
    plt.close()

    report = classification_report(y_true, y_pred)

    with open("results/multimodal_results.txt", "w") as f:
        f.write("Multimodal Model Results\n")
        f.write("========================\n\n")
        f.write("Env features used: " + ", ".join(ENV_COLS) + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    # Save training history plots
    save_history_plot([hist1, hist2], "results/multimodal_training_history.png", "Multimodal Training History")

    # Save model
    model.save("data/multimodal_model.keras")

    print("\n✅ Saved confusion matrix: results/multimodal_confusion_matrix.png")
    print("✅ Saved classification report: results/multimodal_results.txt")
    print("✅ Saved training history plots: results/multimodal_training_history_acc.png & _loss.png")
    print("✅ Saved model: data/multimodal_model.keras")


if __name__ == "__main__":
    main()