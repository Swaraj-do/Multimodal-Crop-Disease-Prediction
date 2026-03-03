import os
import random
import pandas as pd

random.seed(42)

DATASET_PATH = "data/images"
OUTPUT_CSV = "data/multimodal_labels.csv"

# Environmental ranges by broad condition (simple + realistic)
# You can keep these fixed for now.
RANGES = {
    "temperature": (20.0, 35.0),   # °C
    "humidity": (40.0, 95.0),      # %
    "rainfall": (0.0, 20.0),       # mm/day (synthetic)
    "wind_speed": (0.0, 6.0),      # m/s
    "season_code": (0, 3)          # 0..3 (summer/monsoon/winter/spring)
}

def sample_env():
    return {
        "temperature": round(random.uniform(*RANGES["temperature"]), 2),
        "humidity": round(random.uniform(*RANGES["humidity"]), 2),
        "rainfall": round(random.uniform(*RANGES["rainfall"]), 2),
        "wind_speed": round(random.uniform(*RANGES["wind_speed"]), 2),
        "season_code": random.randint(*RANGES["season_code"]),
    }

def main():
    classes = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    class_to_label = {cls: i for i, cls in enumerate(classes)}

    rows = []
    for cls in classes:
        cls_dir = os.path.join(DATASET_PATH, cls)
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(DATASET_PATH, cls, fname).replace("\\", "/")
            env = sample_env()
            rows.append({
                "image_path": img_path,
                **env,
                "label": class_to_label[cls]
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    import json
    with open("data/class_mapping.json", "w") as f:
         json.dump(class_to_label, f, indent=2)
    print("✅ Saved class mapping: data/class_mapping.json")

    print("✅ Created:", OUTPUT_CSV)
    print("Samples:", len(df))
    print("Classes:", len(classes))
    print("\nFirst 5 rows:\n", df.head())

if __name__ == "__main__":
    main()