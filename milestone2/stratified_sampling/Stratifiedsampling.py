
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# Set dataset path
image_dir = r"C:\Users\Reshmi\Capstone-Project\milestone2\Capstone-Project\milestone2\Reshmi_image_stratification\Subgrouped_images"
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
                    'filename': os.path.join(folder_name, filename),  # relative path
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

# Example CSV structure: filename, region, class
# region: 'posterior' or 'anterior'
# class: 'caries' or 'non-caries'

# Load metadata file
df = pd.read_csv('dental_images.csv')

# Create a stratification key
df['stratify_col'] = df['region'] + '_' + df['class']

# Perform stratified split (80% train, 20% test)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['stratify_col'],
    random_state=42
)

# reset indices
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Save the splits
train_df.to_csv('train_split.csv', index=False)
test_df.to_csv('test_split.csv', index=False)

print("Train split:", train_df['stratify_col'].value_counts())
print("Test split:", test_df['stratify_col'].value_counts())

# Load the splits
train_df = pd.read_csv("train_split.csv")
test_df = pd.read_csv("test_split.csv")

# Original image location
base_folder = r"C:\Users\Reshmi\Capstone-Project\milestone2\Capstone-Project\milestone2\Reshmi_image_stratification\Subgrouped_images"

# Output directory
output_dir = "output_images"

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

# Copy train and test images
copy_images(train_df, "train")
copy_images(test_df, "test")

