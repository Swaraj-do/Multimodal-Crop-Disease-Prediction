import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

CSV_PATH = "data/multimodal_labels.csv"
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS_HEAD = 8
EPOCHS_FINE = 4

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def make_ds(df, num_classes, training=True):
    paths = df["image_path"].values
    labels = tf.keras.utils.to_categorical(df["label"].values, num_classes=num_classes)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map(p, y):
        img = load_image(p)
        if training:
            # simple augmentation
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.1)
        return img, y

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2048)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

def build_model(num_classes):
    inp = tf.keras.Input(shape=(*IMG_SIZE, 3))
    base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_tensor=inp)
    base.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base

def main():
    df = pd.read_csv(CSV_PATH)
    num_classes = df["label"].nunique()

    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])
    train_df, val_df  = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df["label"])

    train_ds = make_ds(train_df, num_classes, training=True)
    val_ds   = make_ds(val_df, num_classes, training=False)
    test_ds  = make_ds(test_df, num_classes, training=False)

    model, base = build_model(num_classes)

    print("\n=== Training classifier head (base frozen) ===\n")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

    print("\n=== Fine-tuning last 30 layers of MobileNetV2 ===\n")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE)

    # Evaluate
    y_true, y_pred = [], []
    for x, y in test_ds:
        p = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(p, axis=1))

    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    model.save("data/image_only_model.keras")
    print("\n✅ Saved model: data/image_only_model.keras")

if __name__ == "__main__":
    main()