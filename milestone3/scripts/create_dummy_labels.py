import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
image_folder = os.path.join(base_dir, "data", "raw_images")
label_folder = os.path.join(base_dir, "data", "raw_labels")

os.makedirs(label_folder, exist_ok=True)

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        base = os.path.splitext(filename)[0]
        label_path = os.path.join(label_folder, base + ".txt")
        with open(label_path, "w", newline="\n") as f:
            f.write("0 0.5 0.5 0.3 0.3\n")  # A correct YOLO label with newline
print("✅ Clean label files written.")
