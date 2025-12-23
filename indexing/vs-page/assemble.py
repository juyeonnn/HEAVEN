import cv2
import json
import os
import sys
import math
import shutil
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


from utils import *
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../encode'))
from encoder import create_encoder

class Assemble:
    """Document Assembly class for creating visual search page (vs-page) images and chunk mappings."""
    
    # Class labels for layout elements
    CLASS_NAMES = {
        0: 'title',
        1: 'plain_text',
        2: 'abandon',
        3: 'figure',
        4: 'figure_caption',
        5: 'table',
        6: 'table_caption',
        7: 'table_footnote',
        8: 'isolate_formula',
        9: 'formula_caption'
    }
    
    def __init__(self, dataset: str, layout_fname: str = "layout"):
        """
        Initialize the Assemble class.
        
        Args:
            dataset: Name of the dataset in /data/HEAVEN/benchmark/
            layout_fname: Name of the layout JSON file (without extension)
        """
        self.dataset = dataset
        self.layout_fname = layout_fname
        self.dataset_path = f"/data/HEAVEN/benchmark/{dataset}"
        self.pages_dir = f"{self.dataset_path}/pages"
        self.layout_path = f"{self.dataset_path}/{layout_fname}.json"
        
        # Load layout data
        if not os.path.exists(self.layout_path):
            raise FileNotFoundError(f"Layout file not found: {self.layout_path}")
        
        with open(self.layout_path, "r") as f:
            self.layout = json.load(f)
        
        # Get split key for this dataset
        self.split_key = '_'
        
        # Get pages and document mapping
        pages = os.listdir(self.pages_dir)
        self.pages = filter_files(pages)
        self.document_dict = get_doc_mapping_layout(self.pages, split_key=self.split_key)
        
        print(f"Initialized Assemble for {self.dataset}")
        print(f"  Pages: {len(self.pages)}")
        print(f"  Documents: {len(self.document_dict)}")
    

    def make_class_mask(
        self,
        page_bboxes: List[Dict],
        target_classes: List[int]
    ) -> List[bool]:
        """
        Create a boolean mask for bboxes based on their class.
        
        Args:
            page_bboxes: List of bbox dictionaries with 'cls' key
            target_classes: List of class to include
            
        Returns:
            Boolean mask indicating which bboxes to include
        """
        cls = [b['cls'] for b in page_bboxes]
        cls_mask = [True if c in target_classes else False for c in cls]
        return cls_mask

    def get_title_regions(
        self,
        page_images: List[str],
        bboxes: List[List[float]],
        bbox_cls_masks: List[List[bool]]
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Extract title regions from pages based on bounding boxes and class masks.
        
        Args:
            page_images: List of paths to page images
            bboxes: List of normalized bounding boxes for each page
            bbox_cls_masks: List of boolean masks indicating which bboxes to extract
            
        Returns:
            Tuple of (title_regions, page_sources) where page_sources indicates which page each region came from
        """
        title_regions = []
        page_sources = []
        
        for page_num, (screenshot_path, page_bboxes, bbox_cls_mask) in enumerate(
            zip(page_images, bboxes, bbox_cls_masks)
        ):
            img = cv2.imread(screenshot_path)
            if img is None:
                print(f"Could not load image: {screenshot_path}")
                continue
            
            height, width = img.shape[:2]
            
            for bbox, cls_mask in zip(page_bboxes, bbox_cls_mask):
                if not cls_mask:
                    continue
                
                # Convert normalized coordinates to pixel coordinates
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
                
                region = img[y1:y2, x1:x2]
                
                # Rotate tall narrow regions (likely rotated text)
                if region.shape[0] > region.shape[1] * 5:
                    region = np.rot90(region, k=-1)
                
                title_regions.append(region)
                page_sources.append(page_num)
        
        return title_regions, page_sources
    
    def gen_vs_page(
        self,
        title_regions: List[np.ndarray],
        output_path: str
    ) -> None:
        """
        Generate a vs-page image from title regions using padding method.
        Stacks regions vertically with centered alignment.
        
        Args:
            title_regions: List of title region images
            output_path: Path to save the vs-page image
        """
        if not title_regions:
            return
        
        # Stack vertically with centered alignment
        max_width = max(region.shape[1] for region in title_regions)
        padded_regions = []
        
        for region in title_regions:
            # Create white canvas with max width
            padded = np.ones((region.shape[0], max_width, 3), dtype=np.uint8) * 255
            # Center the region horizontally
            x_offset = (max_width - region.shape[1]) // 2
            padded[:, x_offset:x_offset + region.shape[1]] = region
            padded_regions.append(padded)
        
        # Stack all regions vertically
        vs_page = np.vstack(padded_regions)
        cv2.imwrite(output_path, vs_page)

    @staticmethod
    def concat_pages(page_image_paths: List[str], output_path: str, min_threshold: int = 3) -> Optional[np.ndarray]:
        """
        Concatenate multiple page images.
        
        Args:
            page_image_paths: List of paths to page images
            output_path: Path to save concatenated image
            min_threshold: Threshold to use grid layout instead of linear concatenation
            
        Returns:
            Concatenated image or None if loading failed
        """
        page_imgs = []
        for page_path in page_image_paths:
            img = cv2.imread(page_path)
            if img is not None:
                page_imgs.append(img)
            else:
                print(f"Warning: Could not load image {page_path}")
        
        if not page_imgs:
            return None
        
        # Use grid concat for many images
        if len(page_imgs) > min_threshold:
            return grid_concat(page_imgs, output_path)
        else:
            return naive_concat(page_imgs, output_path)
    
    def construct_vs_page(
        self,
        page_images: List[str],
        page_names: List[str],
        bboxes: List[List[float]],
        bbox_cls_masks: List[List[bool]],
        output_path: str,
        region_min_threshold: int = 4,
        reduction_factor: int = 20,
        chunk_threshold: int = 20,
        page_len: int = 0
    ) -> Dict[str, List[str]]:
        """
        Extract title regions and create vs-page images with chunking.
        Stacks regions vertically with centered alignment.
        
        Args:
            page_images: List of paths to page images
            page_names: List of page names
            bboxes: List of normalized bounding boxes
            bbox_cls_masks: List of boolean masks for filtering bboxes
            output_path: Base path for output files
            region_min_threshold: Minimum number of regions to create chunks
            reduction_factor: Target reduction factor for pages
            chunk_threshold: Page threshold for creating chunks
            page_len: Total number of pages in document
            
        Returns:
            Dictionary mapping chunk names to page names
        """
        mapping = {}
        title_regions, page_sources = self.get_title_regions(page_images, bboxes, bbox_cls_masks)
        page_source_names = [page_names[i] for i in page_sources]
        
        # Extract base filename
        base_filename = output_path.replace('_vs_page.png', '')
        filename_base = base_filename.split('/')[-1]
        
        # Case1: If too few regions, concatenate entire pages
        if len(title_regions) < region_min_threshold:
            fname = f"{base_filename}_chunk0_vs_page.png"
            self.concat_pages(page_images, fname)
            mapping[filename_base + "_chunk0_vs_page"] = list(set(page_source_names))
            return mapping
        
        # Case2: If document is small, create single vs-page
        if page_len <= chunk_threshold:
            fname = f"{base_filename}_chunk0_vs_page.png"
            self.gen_vs_page(title_regions, fname)
            mapping[filename_base + "_chunk0_vs_page"] = list(set(page_source_names))
            return mapping
        
        # Case3-1: Create chunks
        region_per_page = math.ceil(len(title_regions) * (reduction_factor / page_len))
        num_chunks = len(title_regions) // region_per_page
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * region_per_page
            end_idx = start_idx + region_per_page
            
            title_regions_chunk = title_regions[start_idx:end_idx]
            source_names = set(page_source_names[start_idx:end_idx])
            
            fname = f"{base_filename}_chunk{chunk_idx}_vs_page.png"
            self.gen_vs_page(title_regions_chunk, fname)
            mapping[filename_base + f"_chunk{chunk_idx}_vs_page"] = list(source_names)
        
        # Handle remaining regions
        if len(title_regions) % region_per_page != 0:
            start_idx = num_chunks * region_per_page
            end_idx = len(title_regions)
            
            title_regions_chunk = title_regions[start_idx:end_idx]
            source_names = set(page_source_names[start_idx:end_idx])
            
            fname = f"{base_filename}_chunk{num_chunks}_vs_page.png"
            self.gen_vs_page(title_regions_chunk, fname)
            mapping[filename_base + f"_chunk{num_chunks}_vs_page"] = list(source_names)
        
        return mapping
    
    def process(
        self,
        output_dir: str = "vs-page",
        reduction_factor: int = 15,
        chunk_threshold: int = 20,
        target_classes: Optional[List[int]] = None
    ) -> Dict[str, List[str]]:
        """
        Process all documents to create vs-page images and chunk mappings.
        Uses padding method for concatenation.
        
        Args:
            output_dir: Directory name for output (within dataset dataset)
            reduction_factor: Target reduction factor for pages
            chunk_threshold: Minimum pages before chunking
            target_classes: List of class IDs to extract (defaults to title classes)
            
        Returns:
            Dictionary mapping chunk names to page names
        """
        # Default to title classes
        if target_classes is None:
            CLASS_NAMES_REV = {v: k for k, v in self.CLASS_NAMES.items()}
            target_classes = [CLASS_NAMES_REV['title']]
        
        # Setup output directory
        output_path = f"{self.dataset_path}/{output_dir}"
        shutil.rmtree(output_path, ignore_errors=True)
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f"{self.dataset_path}/chunk_mapping", exist_ok=True)
        
        print(f"Processing vs-page creation...")
        print(f"  Target classes: {target_classes} {[self.CLASS_NAMES[c] for c in target_classes]}")
        print(f"  Results will be saved in {output_path}")
        
        no_text_cnt = 0
        chunk_mappings = {}
        total_bbox_num = 0
        
        # Process each document
        for doc, pages in tqdm(self.document_dict.items()):
            page_len = len(pages)
            images = []
            image_names = []
            bboxes = []
            bbox_cls_masks = []
            
            output_file = os.path.join(output_path, f"{doc}_vs_page.png")
            
            for page in pages:
                normalized_page = clean_name(page)
                image_names.append(normalized_page)
                
                if normalized_page not in self.layout:
                    print(f"Page {page} not in layout")
                    no_text_cnt += 1
                    continue
                
                page_bboxes = self.layout[normalized_page]['bbox']
                
                image_path = f"{self.pages_dir}/{page}"
                images.append(image_path)
                bboxes.append([b['xyxyn'] for b in page_bboxes])
                
                cls_mask = self.make_class_mask(page_bboxes, target_classes)
                bbox_cls_masks.append(cls_mask)
                total_bbox_num += len(page_bboxes)
            
            # Extract and create vs-page
            mapping = self.construct_vs_page(
                page_images=images,
                page_names=image_names,
                bboxes=bboxes,
                bbox_cls_masks=bbox_cls_masks,
                output_path=output_file,
                page_len=page_len,
                reduction_factor=reduction_factor,
                chunk_threshold=chunk_threshold
            )
            chunk_mappings.update(mapping)
        
        # Save chunk mappings
        json_path = f"{self.dataset_path}/{self.layout_fname}_chunk_mapping.json"
        with open(json_path, "w") as f:
            json.dump(chunk_mappings, f)
        
        print(f"\nProcessing complete:")
        print(f"  Total bboxes: {total_bbox_num}")
        print(f"  Chunks created: {len(chunk_mappings)}")
        print(f"  Pages not in layout: {no_text_cnt}")
        print(f"  Average pages per chunk: {len(self.pages) / len(chunk_mappings):.2f}")
        print(f"  Saved chunk mappings to {json_path}")
        
        return chunk_mappings
    
    def postprocess(
        self,
        reduction_factor: int = 15
    ) -> Dict[str, List[str]]:
        """
        Post-process chunk mappings to ensure all pages are included.
        
        This fills in any gaps in the chunk mappings by distributing pages
        from the document mapping into the chunks.
        
        Args:
            reduction_factor: Reduction factor used during vs-page creation
            
        Returns:
            Updated chunk mapping dictionary
        """
        # Create document mapping with cleaned names
        document_mapping = {}
        for doc_name, page_lst in self.document_dict.items():
            document_mapping[doc_name] = [clean_name(p) for p in page_lst]
        
        # Load existing chunk mapping
        chunk_mapping_path = f"{self.dataset_path}/{self.layout_fname}_chunk_mapping.json"
        with open(chunk_mapping_path, "r") as f:
            chunk_mapping = json.load(f)
        
        print(f"Post-processing chunk mappings...")
        print(f"  Documents: {len(document_mapping)}, VS-page chunks: {len(chunk_mapping)}")
        
        # Sort chunk pages by page number
        for chunk_name, page_lst in chunk_mapping.items():
            page_nums = []
            for p in page_lst:
                _, page_num = split_page_name(p, self.split_key)
                page_nums.append(page_num)
            chunk_mapping[chunk_name] = sort_doc_by_page(page_lst, page_nums)
        
        # Distribute all pages into chunks
        for doc_name, page_lst in document_mapping.items():
            remaining_pages = page_lst.copy()
            chunk_idx = 0
            
            while True:
                chunk_key = f"{doc_name}_chunk{chunk_idx}_vs_page"
                
                # Check if this chunk exists
                if chunk_key not in chunk_mapping:
                    # No more chunks - add remaining pages to previous chunk
                    if chunk_idx > 0:
                        last_chunk_key = f"{doc_name}_chunk{chunk_idx-1}_vs_page"
                        chunk_mapping[last_chunk_key].extend(remaining_pages)
                        remaining_pages = []
                    break
                
                chunk_idx += 1
                chunk_pages = chunk_mapping[chunk_key]
                
                # Handle empty chunks
                if not chunk_pages:
                    chunk_mapping[chunk_key] = remaining_pages
                    remaining_pages = []
                    break
                
                last_page = chunk_pages[-1]
                
                # Find position in remaining pages
                if last_page in remaining_pages:
                    last_page_index = remaining_pages.index(last_page)
                    
                    # Add pages up to the last page
                    chunk_mapping[chunk_key].extend(remaining_pages[:last_page_index])
                    remaining_pages = remaining_pages[last_page_index:]
        
        # Sort and deduplicate
        for chunk_name, page_lst in chunk_mapping.items():
            page_lst = list(set(page_lst))
            page_nums = []
            for p in page_lst:
                _, page_num = split_page_name(p, self.split_key)
                page_nums.append(page_num)
            chunk_mapping[chunk_name] = sort_doc_by_page(page_lst, page_nums)
        
        # Verification
        all_doc_pages = set()
        all_chunk_pages = set()
        
        for pages in document_mapping.values():
            all_doc_pages.update(pages)
        
        for pages in chunk_mapping.values():
            all_chunk_pages.update(pages)
        
        print(f"\nPost-processing complete:")
        print(f"  Unique pages: {len(all_doc_pages)} == Unique pages from VS-page chunks: {len(all_chunk_pages)}")
        
        # Save processed mapping
        output_path = f"{self.dataset_path}/{self.layout_fname}_chunk_mapping_postprocessed.json"
        with open(output_path, "w") as f:
            json.dump(chunk_mapping, f, indent=4)
        
        print(f"  Saved to {output_path}")
        
        return chunk_mapping


def main():
    """Command-line interface for Assemble."""
    parser = argparse.ArgumentParser(description="Document Assembly - Create vs-page images and chunk mappings")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Folder name in /data/HEAVEN/benchmark/")
    parser.add_argument("--layout_fname", type=str, default="layout",
                       help="Layout JSON filename (without extension)")
    parser.add_argument("--reduction_factor", type=int, default=15,
                       help="Target reduction factor for pages")
    parser.add_argument("--chunk_threshold", type=int, default=20,
                       help="Minimum pages before chunking")
    parser.add_argument("--output_dir", type=str, default="vs-page",
                       help="Output directory name")
    parser.add_argument("--encoder_type", type=str, default="dse",
                       help="Type of encoder to use")
    
    args = parser.parse_args()
    
    # Initialize Assemble
    assembler = Assemble(dataset=args.dataset, layout_fname=args.layout_fname)
    
    # Process vs-page
    print("\n=== Step 1: Creating vs-page images ===")
    chunk_mappings = assembler.process(
        output_dir=args.output_dir,
        reduction_factor=args.reduction_factor,
        chunk_threshold=args.chunk_threshold
    )
    
    # Post-process chunk mappings 
    print("\n=== Step 2: Post-processing chunk mappings ===")
    chunk_mappings = assembler.postprocess(reduction_factor=args.reduction_factor)
    
    print("\n=== Assembly Complete ===")
    print(f"Total VS-page chunks created: {len(chunk_mappings)}")

    print("\n=== Step 3: Encoding vs-page images ===")
    encoder = create_encoder(args.encoder_type)
    encoder.process_vs_page(encoder_type=args.encoder_type, dataset=args.dataset)
    encoder.cleanup()


if __name__ == "__main__":
    main()

