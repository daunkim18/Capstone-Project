import os, shutil, random, yaml
from tqdm import tqdm

def load_cfg(path="C:/Users/daunk/Capstone-Project/milestone3/configs/data.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def stratified_split(src_img, src_lbl, dst, seed=42):
    print("Image source folder:", src_img)
    print("Label source folder:", src_lbl)

    imgs = [f for f in sorted(os.listdir(src_img)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print("Found", len(imgs), "image(s) before split.")
    print("Example image file(s):", imgs[:3])

    random.seed(seed)
    random.shuffle(imgs)
    n = len(imgs)
    train_val = imgs[:int(n*0.8)]
    test = imgs[int(n*0.8):]
    train = train_val[:int(len(train_val)*0.75)]
    val = train_val[int(len(train_val)*0.75):]

    for split, files in {'train': train, 'val': val, 'test': test}.items():
        for f in tqdm(files, desc=split):
            base = f.rsplit('.',1)[0]
            label_path = os.path.join(src_lbl, base + '.txt')
            image_path = os.path.join(src_img, f)

            if os.path.exists(label_path) and os.path.exists(image_path):
                os.makedirs(os.path.join(dst, 'images', split), exist_ok=True)
                os.makedirs(os.path.join(dst, 'labels', split), exist_ok=True)
                shutil.copy(image_path, os.path.join(dst, 'images', split, f))
                shutil.copy(label_path, os.path.join(dst, 'labels', split, base + '.txt'))
            else:
                print(f"Missing pair for: {f} (image or label)")

if __name__=='__main__':
    cfg = load_cfg()
    img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw_images"))
    lbl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw_labels"))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    print("img_dir:", img_dir)
    print("lbl_dir:", lbl_dir)
    print("output_dir:", output_dir)
    print("Files in image folder:", os.listdir(img_dir))
    print("Files in label folder:", os.listdir(lbl_dir))
    stratified_split(img_dir, lbl_dir, output_dir)

