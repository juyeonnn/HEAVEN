import pytesseract
import os
from tqdm import tqdm
from PIL import Image
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ViMDoc")
parser.add_argument("--max_workers", type=int, default=16, help="Number of worker processes (default: CPU count)")
args = parser.parse_args()

def clean(name):
    return name.replace('.png','').replace('.jpg','').replace('.jpeg','')

def filter_files(files):
    return [f for f in files if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

def process_single_image(file_info):
    """Process a single image file and return OCR result"""
    folder_path, file = file_info
    try:
        file_path = os.path.join(folder_path, file)
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return {
            "image": file,
            "text": text
        }
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return {
            "image": file,
            "text": ""
        }

def main():
    json_path = f"/mnt/HEAVEN/benchmark/{args.dataset}/ocr.json"
    folder_path = f'/mnt/HEAVEN/benchmark/{args.dataset}/pages'
    
    # Get all image files
    files = sorted(os.listdir(folder_path))
    files = filter_files(files)
    
    print(f"Processing {len(files)} images")
    
    # Determine number of workers
    max_workers = args.max_workers or cpu_count()
    print(f"Using {max_workers} worker processes")
    
    # Prepare file info for multiprocessing
    file_infos = [(folder_path, file) for file in files]
    
    converted_text = []
    
    # Process images in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_image, file_info): file_info[1] 
                         for file_info in file_infos}
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_file), total=len(files), desc="OCR Progress"):
            file = future_to_file[future]
            try:
                result = future.result()
                converted_text.append(result)
            except Exception as e:
                print(f"Exception processing {file}: {e}")
                converted_text.append({
                    "image": file,
                    "text": ""
                })
    
    # Sort results by filename to maintain consistent order
    converted_text.sort(key=lambda x: x["image"])
    
    # Save results
    with open(json_path, "w") as f:
        json.dump(converted_text, f, ensure_ascii=False, indent=2)
    print(f"Saved converted text to {json_path}")

if __name__ == "__main__":
    main()