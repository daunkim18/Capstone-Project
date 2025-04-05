
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import glob
import matplotlib.pyplot as plt

# Set dataset path
image_dir = r"C:\Users\Reshmi\Capstone-Project\milestone2\Capstone-Project\milestone2\stratified_sampling\Subgrouped_images"
data = []

for folder_name in os.listdir(image_dir):
    folder_path = os.path.join(image_dir, folder_name)
    if os.path.isdir(folder_path):
        try:
            region, label = folder_name.split('_')
        except ValueError:
            print(f"Skipping unexpected folder format: {folder_name}")
            continue

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                data.append({
                    'filename': os.path.join(folder_name, filename),
                    'region': region.lower(),
                    'class': label.lower()
                })

# Create DataFrame
df = pd.DataFrame(data)

if df.empty:
    print("No images found")
else:
    df.to_csv("dental_images.csv", index=False)
    print(f"CSV created with {len(df)} rows: dental_images.csv")
    print(df.head())

# Load data
df = pd.read_csv("dental_images.csv")
df['stratify_col'] = df['region'] + '_' + df['class']

# Step 1: Split into train+val and test (80% train_val, 20% test)
train_val_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['stratify_col'],
    random_state=42
)

# Step 2: Split train+val into train and val (75% train, 25% val of train_val)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.25,
    stratify=train_val_df['stratify_col'],
    random_state=42
)

# Reset indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Save splits
train_df.to_csv('train_split.csv', index=False)
val_df.to_csv('val_split.csv', index=False)
test_df.to_csv('test_split.csv', index=False)

# Print distribution counts
print("\nTrain distribution:\n", train_df['stratify_col'].value_counts())
print("\nValidation distribution:\n", val_df['stratify_col'].value_counts())
print("\nTest distribution:\n", test_df['stratify_col'].value_counts())

# Output directory
output_dir = "output_images"
base_folder = image_dir

# Clear existing output directory if exists
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Function to copy images
def copy_images(df, split_type):
    for _, row in df.iterrows():
        subfolder = f"{row['region']}_{row['class']}"
        base_path = os.path.join(base_folder, row['filename'])
        dest_path = os.path.join(output_dir, split_type, subfolder)
        os.makedirs(dest_path, exist_ok=True)
        try:
            shutil.copy(base_path, dest_path)
        except FileNotFoundError:
            print(f"File not found: {base_path}")

# Copy all images
copy_images(train_df, "train")
copy_images(val_df, "val")
copy_images(test_df, "test")

# Print how many images were copied
train_files = glob.glob(os.path.join(output_dir, "train", "*", "*"))
val_files = glob.glob(os.path.join(output_dir, "val", "*", "*"))
test_files = glob.glob(os.path.join(output_dir, "test", "*", "*"))

print(f"\nImages copied:")
print(f"Train: {len(train_files)}")
print(f"Validation: {len(val_files)}")
print(f"Test: {len(test_files)}")

# Visual check with bar plots
train_df['stratify_col'].value_counts().plot(kind='bar', title='Train Distribution')
plt.show()
val_df['stratify_col'].value_counts().plot(kind='bar', title='Validation Distribution')
plt.show()
test_df['stratify_col'].value_counts().plot(kind='bar', title='Test Distribution')
plt.show()


