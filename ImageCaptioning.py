import torch
import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", local_files_only=True)
model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def getCaption(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)


import os
import shutil

def create_metadata_db(images_dir="images", output_file="metadata.json"):
    """Creates a JSON metadata database of the images directory."""

    metadata = []
    if os.path.exists(images_dir):
        # Sort files to ensure deterministic ordering of embeddings, matching load_captions()
        filenames = sorted(os.listdir(images_dir))
        idx = 0
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                name, _ = os.path.splitext(filename)
                caption = getCaption(os.path.join(images_dir,filename))
                image_path = os.path.join(images_dir, filename)
                metadata.append({
                    "id": idx,
                    "file_name": filename,
                    "image_path": image_path,
                    "caption": caption
                })
                idx += 1
                
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Created metadata database with {len(metadata)} entries saved to {output_file}")
    return metadata


#dont run this portion, it's just for testing
def __caption_images(dirctory_path="/home/arya/Desktop/share/", dest_dir="images") -> None:
    os.makedirs(dest_dir, exist_ok=True)
    images = os.listdir(dirctory_path)
    for image_name in images:
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            src_path = os.path.join(dirctory_path, image_name)
            caption_text = getCaption(src_path)
            
            # Sanitize caption for a valid filename
            safe_name = "".join(c for c in caption_text if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_name = " ".join(safe_name.split()) # normalize whitespace
            
            _, ext = os.path.splitext(image_name)
            new_name = f"{safe_name}{ext}"
            dest_path = os.path.join(dest_dir, new_name)
            
            # Handle duplicate names by appending a suffix
            counter = 1
            base_name = safe_name
            while os.path.exists(dest_path):
                new_name = f"{base_name}_{counter}{ext}"
                dest_path = os.path.join(dest_dir, new_name)
                counter += 1
                
            print(f"Copying: {image_name} -> {new_name}")
            shutil.copy2(src_path, dest_path)


if __name__=="__main__":
    # while True:
    #     image_path = input("Enter an image path (or 'exit' to quit): ")
    #     if image_path.lower() == 'exit':
    #         break
    #     print(getCaption(image_path))
    create_metadata_db("/home/arya/Desktop/share/")
    