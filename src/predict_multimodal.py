import os
import json
import numpy as np
import tensorflow as tf

from risk_and_reco import ENV_COLS, env_influence_score, risk_level, recommendations

MODEL_PATH = "data/multimodal_model.keras"
CLASS_MAP_PATH = "data/class_mapping.json"
IMG_SIZE = (224, 224)


def load_image_from_path(path: str) -> np.ndarray:
    img_bytes = tf.io.read_file(path)
    lower = tf.strings.lower(path)

    def decode_png():
        return tf.image.decode_png(img_bytes, channels=3)

    def decode_jpeg():
        return tf.image.decode_jpeg(img_bytes, channels=3)

    img = tf.cond(
        tf.strings.regex_full_match(lower, r".*\.png"),
        true_fn=decode_png,
        false_fn=decode_jpeg
    )

    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()


def invert_mapping(mapping: dict) -> dict:
    return {int(v): k for k, v in mapping.items()}


def get_user_env_input() -> dict:
    print("\nEnter environmental values:")
    temperature = float(input("Temperature: "))
    humidity = float(input("Humidity: "))
    rainfall = float(input("Rainfall: "))
    wind_speed = float(input("Wind Speed: "))
    season_code = int(input("Season Code [0=Summer, 1=Monsoon, 2=Winter, 3=Spring]: "))

    return {
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "wind_speed": wind_speed,
        "season_code": season_code
    }


def main():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_MAP_PATH, "r") as f:
        class_map = json.load(f)

    inv_map = invert_mapping(class_map)

    image_path = input("\nEnter image path: ").strip()
    env = get_user_env_input()

    img = load_image_from_path(image_path)
    img = np.expand_dims(img, axis=0)

    env_vec = np.array([[env[c] for c in ENV_COLS]], dtype=np.float32)

    probs = model.predict({"img": img, "env": env_vec}, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_prob = float(np.max(probs))
    pred_class = inv_map[pred_idx]

    env_score = env_influence_score(env)
    risk = risk_level(pred_prob, env_score)
    recs = recommendations(pred_class, risk, env)

    # Top predictions
    top_idx = np.argsort(probs)[::-1][:3]

    # Build output text
    output_lines = []
    output_lines.append("=" * 60)
    output_lines.append("MULTIMODAL CROP DISEASE PREDICTION RESULT")
    output_lines.append("=" * 60)
    output_lines.append(f"Predicted Disease        : {pred_class}")
    output_lines.append(f"Disease Probability      : {pred_prob * 100:.2f}%")
    output_lines.append(f"Environmental Influence  : {env_score:.3f}")
    output_lines.append(f"Risk Level               : {risk}")

    output_lines.append("\nTop Predictions:")
    for i, idx in enumerate(top_idx, start=1):
        output_lines.append(f"{i}. {inv_map[idx]} -> {probs[idx] * 100:.2f}%")

    output_lines.append("\nRecommendations:")
    for i, rec in enumerate(recs, start=1):
        output_lines.append(f"{i}. {rec}")

    output_lines.append("=" * 60)

    output_text = "\n".join(output_lines)

    # Print in terminal
    print("\n" + output_text)

    # Save automatically
    os.makedirs("results", exist_ok=True)
    with open("results/sample_prediction_output.txt", "w", encoding="utf-8") as f:
        f.write(output_text)

    print("\n Output saved to results/sample_prediction_output.txt")


if __name__ == "__main__":
    main()