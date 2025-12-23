"""
Utility functions for document processing and page management.
"""

import os
import re
from typing import List, Dict, Tuple, Optional, Any
import json
from pathlib import Path
import math
import numpy as np
import cv2


def clean_name(fname: str) -> str:
    """Remove image extensions from filename."""
    return fname.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')


def grid_concat(images: List[np.ndarray], output_path: str, padding: int = 5) -> np.ndarray:
    """
    Create a grid layout for multiple images.
        
    Returns:
        The concatenated image
    """
    num_images = len(images)
    grid_size = int(math.ceil(math.sqrt(num_images)))
    grid_rows = grid_size
    grid_cols = grid_size
    
    # Adjust grid to be more rectangular if needed
    if grid_size * (grid_size - 1) >= num_images:
        grid_rows = grid_size - 1
    
    # Calculate average aspect ratio
    aspect_ratios = [img.shape[1] / img.shape[0] for img in images]
    avg_aspect = sum(aspect_ratios) / len(aspect_ratios)
    
    # Calculate target cell size
    total_area = sum(img.shape[0] * img.shape[1] for img in images)
    avg_area = total_area / num_images
    target_height = int(math.sqrt(avg_area / avg_aspect))
    target_width = int(avg_aspect * target_height)
    
    # Resize all images
    resized_images = []
    for img in images:
        h, w = img.shape[:2]
        aspect = w / h
        
        if aspect > avg_aspect:
            new_width = target_width
            new_height = int(new_width / aspect)
        else:
            new_height = target_height
            new_width = int(new_height * aspect)
        
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_images.append(resized)
    
    # Create canvas
    canvas_width = grid_cols * target_width + (grid_cols - 1) * padding
    canvas_height = grid_rows * target_height + (grid_rows - 1) * padding
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # Place images on grid
    for idx, img in enumerate(resized_images):
        if idx >= num_images:
            break
        
        row = idx // grid_cols
        col = idx % grid_cols
        
        y_start = row * (target_height + padding)
        x_start = col * (target_width + padding)
        
        h, w = img.shape[:2]
        y_offset = (target_height - h) // 2
        x_offset = (target_width - w) // 2
        
        canvas[y_start + y_offset:y_start + y_offset + h,
                x_start + x_offset:x_start + x_offset + w] = img
    
    cv2.imwrite(output_path, canvas)
    return canvas


def naive_concat(images: List[np.ndarray], output_path: str) -> np.ndarray:
    """
    Concatenate multiple images vertically or horizontally.
    
    Returns:
        The concatenated image
    """
    widths = [img.shape[1] for img in page_imgs]
    heights = [img.shape[0] for img in page_imgs]
    total_width = sum(widths)
    total_height = sum(heights)
    
    # Concatenate vertically
    if total_width > total_height:
        min_width = min(widths)
        resized_imgs = [cv2.resize(img, (min_width, int(img.shape[0] * min_width / img.shape[1]))) 
                        for img in page_imgs]
        concatenated = np.vstack(resized_imgs)
    # Concatenate horizontally
    else:
        min_height = min(heights)
        resized_imgs = [cv2.resize(img, (int(img.shape[1] * min_height / img.shape[0]), min_height)) 
                        for img in page_imgs]
        concatenated = np.hstack(resized_imgs)
    
    cv2.imwrite(output_path, concatenated)
    return concatenated

def filter_files(files: List[str], extensions: Optional[List[str]] = None) -> List[str]:
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    
    filtered = []
    for f in files:
        if any(f.endswith(ext) for ext in extensions):
            # Skip hidden files and system files
            if not f.startswith('.') and not f.startswith('__'):
                filtered.append(f)
    
    return sorted(filtered)


def split_page_name(page_name: str, split_key: str = '_') -> Tuple[str, int]:
    """
    Split a page name into document name and page number.
    
    Args:
        page_name: Full page name (e.g., 'doc1_page_5' or 'doc1_5')
        split_key: Delimiter between doc name and page number
        
    Returns:
        Tuple of (document_name, page_number)
    """
    # Remove any file extensions first
    page_name = clean_name(page_name)
    
    # Split by the key
    if split_key in page_name:
        parts = page_name.split(split_key)
        doc_name = split_key.join(parts[:-1])  # Everything except last part
        try:
            page_num = int(parts[-1])
        except ValueError:
            # If last part is not a number, try to extract number
            page_num = extract_page_number(parts[-1])
    else:
        # Try to extract the last number from the string
        match = re.search(r'(\d+)$', page_name)
        if match:
            page_num = int(match.group(1))
            doc_name = page_name[:match.start()].rstrip('_-')
        else:
            # No page number found
            doc_name = page_name
            page_num = 0
    
    return doc_name, page_num


def get_doc_mapping_layout(
    pages: List[str],
    split_key: str = '_'
) -> Dict[str, List[str]]:
    """
    Create a mapping from document names to their page lists.
    Pages are sorted by page number within each document.
    
    Args:
        pages: List of page filenames
        split_key: Delimiter for splitting document name and page number
        
    Returns:
        Dictionary mapping document names to sorted lists of page filenames
    """
    doc_mapping = {}
    
    for page in pages:
        doc_name, page_num = split_page_name(page, split_key)
        
        if doc_name not in doc_mapping:
            doc_mapping[doc_name] = []
        
        doc_mapping[doc_name].append((page, page_num))
    
    # Sort pages by page number within each document
    for doc_name in doc_mapping:
        # Sort by page number (second element of tuple)
        doc_mapping[doc_name] = [page for page, _ in sorted(doc_mapping[doc_name], key=lambda x: x[1])]
    
    return doc_mapping


def sort_doc_by_page(pages: List[str], page_nums: List[int]) -> List[str]:
    """
    Sort pages by their page numbers.
    
    Args:
        pages: List of page names
        page_nums: Corresponding list of page numbers
        
    Returns:
        Sorted list of page names
    """
    if len(pages) != len(page_nums):
        raise ValueError(f"Length mismatch: pages={len(pages)}, page_nums={len(page_nums)}")
    
    # Combine and sort
    combined = list(zip(pages, page_nums))
    combined.sort(key=lambda x: x[1])
    
    return [page for page, _ in combined]

