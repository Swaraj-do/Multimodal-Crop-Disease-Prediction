import os

DATASET_PATH = "data/images"

classes = os.listdir(DATASET_PATH)

print("Total Classes:", len(classes))
print("\nSample Classes:\n")

for c in classes[:10]:
    print(c)

# count images
total_images = 0

for cls in classes:
    cls_path = os.path.join(DATASET_PATH, cls)
    total_images += len(os.listdir(cls_path))

print("\nTotal Images:", total_images)