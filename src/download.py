    
from datasets import load_dataset, Features, Value, Image
from tqdm import tqdm
import os
from PIL import UnidentifiedImageError

def download_final_evaluation_dataset():
    my_features = Features({
        "image": Image(decode=False),
        "label": Value("string")  # <--- This prevents the ClassLabel error
    })

    # 2. Load the dataset with these overridden features
    dataset = load_dataset(
        "ZihaoZhu/EARDataset",
        features=my_features,
        verification_mode="no_checks"
    )
    output_root = "./EARDataset/images"

    # Loop through the dataset (assuming 'train' split exists)
    # If your dataset doesn't have splits, just use 'dataset' instead of 'dataset["train"]'
    print(f"Exporting images to {output_root}...")

    # Counters for stats
    success_count = 0
    fail_count = 0

    print(f"Exporting images to {output_root}...")

    # Loop through the dataset
    for i, item in tqdm(enumerate(dataset['train'])):
        print("Item: ", item, flush=True)
        if item['image']["bytes"] is None:
            fail_count += 1
            continue
        try:
            # 1. Try to access/decode the image
            # This is usually where UnidentifiedImageError happens
            image = item['image']
            
            # If image is None (sometimes happens with bad decoding), skip it
            if image is None:
                raise UnidentifiedImageError("Image is None")

            # 2. Prepare the label and folder
            # Convert label to string and sanitize it for folder names
            label_raw = item.get('label', 'unknown')
            label = str(label_raw).replace(" ", "_").replace("/", "_")
            
            label_dir = os.path.join(output_root, label)
            os.makedirs(label_dir, exist_ok=True)
            
            # 3. Save the image
            # We convert to RGB to handle potential RGBA/Palette issues
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            save_path = os.path.join(label_dir, f"image_{i}.jpg")
            image.save(save_path)
            
            success_count += 1

        except (UnidentifiedImageError, OSError, ValueError) as e:
            # If the image is corrupt, we catch the error here
            # print(f"Skipping corrupt image at index {i}: {e}") # Uncomment to see details
            fail_count += 1
            continue

    print("--- Export Complete ---")
    print(f"Successfully saved: {success_count} images")
    print(f"Skipped (corrupt):  {fail_count} images")
    print(f"Check the '{output_root}' folder to view your data.")



if __name__ == "__main__":
    download_final_evaluation_dataset()
