"""
GME Encoder implementation.
"""

import torch
from PIL import Image, ImageFile
from transformers import AutoModel
import os
from encoder import Encode
os.environ["TOKENIZERS_PARALLELISM"] = "false"
Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True

class GMEEncoder(Encode):
    """GME encoder implementation."""
    
    def __init__(self,**kwargs):
        """
        Initialize GME encoder.
        """
        super().__init__(**kwargs)
    
    def _load_model(self, **kwargs):
        """Load GME model."""
        self.model = AutoModel.from_pretrained(
            f"Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
            cache_dir="/mnt/transformers-cache",
            dtype="float16", 
            device_map=f'cuda:{self.device}',
            trust_remote_code=True,
            use_fast=True
        ).eval()
    
    def _process_single_item(self, item_path: str):
        """Process a single image and return embedding."""
        image = Image.open(item_path)
        embeddings = self.model.get_image_embeddings(images=[image], is_query=False)
        return embeddings.squeeze(), None
    
    def _process_single_query(self, query: str):
        """Process a single query and return embedding."""
        # Define the instruction prompt for text-to-image retrieval
        t2i_prompt = "Find a screenshot that relevant to the user's question."
        
        with torch.no_grad():
            embedding = self.model.get_text_embeddings(texts=[query], instruction=t2i_prompt)
        
        return embedding, None
