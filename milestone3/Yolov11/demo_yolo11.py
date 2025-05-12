import cv2
import os

# Paths
image_path = 'data/images/test/demo.jpg' 
label_path = 'data/labels/test/demo.txt'  
output_path = 'demo_output.jpg'

# Class labels 
class_names = ['caries(anterior)', 'caries(posterior)', 'caries(mixed)', 'healthy']

# Load image
image = cv2.imread(image_path)
height, width, _ = image.shape

# Draw bounding boxes
with open(label_path, 'r') as f:
    for line in f:
        cls_id, x_center, y_center, w, h = map(float, line.strip().split())
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)

        color = (0, 255, 0) if cls_id == 0 else (255, 0, 0) if cls_id == 1 else (0, 0, 255)
        label = class_names[int(cls_id)]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save result
cv2.imwrite(output_path, image)
print(f"Output saved as {output_path}")
