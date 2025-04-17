import os

img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "images", "train"))
print("Looking in:", img_dir)

if not os.path.exists(img_dir):
    print("❌ Folder not found.")
else:
    print("✅ Folder exists.")
    files = os.listdir(img_dir)
    print("Files in folder:", files)
    
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print("🖼️ Image files detected:", image_files)
