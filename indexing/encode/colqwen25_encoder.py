"""
ColPali2.5 Encoder implementation.
"""

import sys
sys.path.append('/mnt/colpali')

import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from encoder import Encode


class ColQwen25Encoder(Encode):
    """ColQwen2.5 encoder implementation."""
    
    def __init__(self, **kwargs):
        """
        Initialize ColQwen2.5 encoder.
        """
        super().__init__(**kwargs)
    
    def _load_model(self, **kwargs):
        """Load ColQwen2.5 model and processor."""
        self.model = ColQwen2_5.from_pretrained(
            "vidore/colqwen2.5-v0.2",
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{self.device}",
            cache_dir="/mnt/transformers-cache",
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        print(f"Model loaded on {self.device}")
        
        self.processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")
    
    def _process_single_item(self, item_path: str):
        """Process a single image and return embedding and input_ids."""
        img = Image.open(item_path)
        
        # Process single image
        processed_image = self.processor.process_images([img]).to(self.model.device)
        
        with torch.no_grad():
            embedding = self.model(**processed_image)
        
        # Extract input_ids
        input_id = processed_image['input_ids'].cpu()
        
        return embedding.cpu(), input_id
    
    def _process_single_query(self, query: str):
        """Process a single query and return embedding and input_ids."""
        # Process single query
        batch_queries = self.processor.process_queries([query]).to(self.model.device)
        
        with torch.no_grad():
            embedding = self.model(**batch_queries)
        
        # Extract input_ids
        input_id = batch_queries['input_ids'].cpu()
        
        return embedding.cpu(), input_id
    
