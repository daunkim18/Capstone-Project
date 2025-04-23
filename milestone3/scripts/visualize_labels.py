import cv2
import os

# === Paths ===
images_folder = 'data/images/train/'    # your image directory
labels_folder = 'data/labels/train/'    # your label directory
output_folder = 'data/labeled_images/'  # output folder

os.makedirs(output_folder, exist_ok=True)

# === Edit class names here ===
class_names = ['caries(anterior)', 'caries(posterior)', 'healthy']

# === Draw boxes ===
for filename in os.listdir(images_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(images_folder, filename)
        label_path = os.path.join(labels_folder, filename.replace('.jpg', '.txt'))

        image = cv2.imread(image_path)
        height, width, _ = image.shape

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, x_center, y_center, w, h = map(float, line.strip().split())
                    x1 = int((x_center - w/2) * width)
                    y1 = int((y_center - h/2) * height)
                    x2 = int((x_center + w/2) * width)
                    y2 = int((y_center + h/2) * height)

                    color = (0, 255, 0) if cls == 0 else (255, 0, 0) if cls == 1 else (0, 0, 255)
                    label = class_names[int(cls)]
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imwrite(os.path.join(output_folder, filename), image)

print("✅ Finished labeled .jpg files saved in:", output_folder)
