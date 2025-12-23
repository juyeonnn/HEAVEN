from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
import cv2
import os 
import json
from tqdm import tqdm
import math
import argparse
from typing import Dict, Optional, List
from utils import *

class DLA:
    """Document Layout Analysis class using DocLayout-YOLO model."""
    
    # Class labels mapping
    CLASS_NAMES = {
        0: "Caption",
        1: "Footnote",
        2: "Formula",
        3: "List-item",
        4: "Page-footer",
        5: "Page-header",
        6: "Picture",
        7: "Section-header",
        8: "Table",
        9: "Text",
        10: "Title"
    }
    
    def __init__(self, device: str = "0", model_path: Optional[str] = None):
        """
        Initialize the DLA model.
        
        Args:
            device: CUDA device ID (e.g., "0", "1") or "cpu"
            model_path: Optional path to model file. If None, downloads from HuggingFace.
        """
        self.device = device
        if model_path is None:
            model_path = hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                filename="doclayout_yolo_docstructbench_imgsz1024.pt"
            )
        self.model = YOLOv10(model_path)
        print(f"DLA model loaded on device: {device}")



    def get_layout(self, image_path: str, imgsz: int = 1024, conf: float = 0.2) -> Dict:
        """
        Predict layout for a single image.
        
        Args:
            image_path: Path to the image file
            imgsz: Prediction image size
            conf: Confidence threshold
            
        Returns:
            Dictionary containing original shape and bounding boxes with layout information
        """
        output = self.model.predict(
            image_path,
            imgsz=imgsz,
            conf=conf,
            device=f"cuda:{self.device}",
            verbose=False
        )
        output = output[0]
        
        ret = {"orig_shape": output.boxes.orig_shape, "bbox": []}
        for o in output:
            for cls, conf, xywh, xywhn, xyxy, xyxyn in zip(
                o.boxes.cls, o.boxes.conf, o.boxes.xywh, 
                o.boxes.xywhn, o.boxes.xyxy, o.boxes.xyxyn
            ):
                ret["bbox"].append({
                    "cls": cls.cpu().tolist(),
                    # "cls_name": self.CLASS_NAMES.get(int(cls.cpu().tolist()), "Unknown"),
                    "conf": conf.cpu().tolist(),
                    "xyxyn": xyxyn.cpu().tolist()
                })
        
        return ret
    
    def process_dataset(
        self, 
        dataset: str, 
        output_json: Optional[str] = None,
        imgsz: int = 1024,
        conf: float = 0.2
    ) -> Dict:
        """
        Process all images in a dataset.
        
        Args:
            dataset: Name of the dataset in /data/HEAVEN/benchmark/
            output_json: Path to save JSON results. If None, uses default location.
            imgsz: Prediction image size
            conf: Confidence threshold
            
        Returns:
            Dictionary with all layout predictions
        """
        print(f"Processing {dataset}")
        page_dir = f"/data/HEAVEN/benchmark/{dataset}/pages"
        
        if not os.path.exists(page_dir):
            raise ValueError(f"Page directory not found: {page_dir}")
        
        # Setup output directories
        layout_dir = f"/data/HEAVEN/benchmark/{dataset}/pages-layout"
        if not os.path.exists(layout_dir):
            os.makedirs(layout_dir)
        
        if output_json is None:
            output_json = f"/data/HEAVEN/benchmark/{dataset}/layout.json"
        
        # Process images
        error_cnt = 0
        data = {}
        files = os.listdir(page_dir)
        files = filter_files(files)

        for fname in tqdm(files):
            key = clean_name(fname)
            try:
                image_path = f"{page_dir}/{fname}"
                ret = self.get_layout(image_path, imgsz=imgsz, conf=conf)
                data[key] = ret
                
            except Exception as e:
                print(f"Error in {fname}: {str(e)}")    
                data[key] = {}
                error_cnt += 1
                continue
        
        # Save results
        with open(output_json, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Error count: {error_cnt}")
        
        print(f"Saved layout to {output_json}")

        return data


def main():
    """Command-line interface for DLA."""
    parser = argparse.ArgumentParser(description="Document Layout Analysis")
    parser.add_argument("--dataset", type=str, default="ViDoSeek", 
                       help="Dataset name in /data/HEAVEN/benchmark/")
    parser.add_argument("--device", type=str, default="0",
                       help="CUDA device ID or 'cpu'")
    parser.add_argument("--imgsz", type=int, default=1024,
                       help="Prediction image size")
    parser.add_argument("--conf", type=float, default=0.2,
                       help="Confidence threshold")
    args = parser.parse_args()
    
    # Initialize and run DLA
    dla = DLA(device=args.device)
    dla.process_dataset(
        dataset=args.dataset,
        imgsz=args.imgsz,
        conf=args.conf
    )
    print("\n=== DLA Complete ===")


if __name__ == "__main__":
    main()
