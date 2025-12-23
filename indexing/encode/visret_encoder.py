"""
VisRet Encoder implementation.
"""

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from transformers import AutoModel, AutoTokenizer
from encoder import Encode
import numpy as np
Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VisRetEncoder(Encode):
    """VisRet encoder implementation."""
    
    def __init__(self,**kwargs):
        """
        Initialize VisRet encoder.
        
        Args:
            device: CUDA device to use
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
    
    def _load_model(self, **kwargs):
        """Load VisRet model and tokenizer."""
        model_name_or_path = "openbmb/VisRAG-Ret"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            cache_dir="/mnt/transformers-cache"
        ).cuda()
        self.model.eval()
    
    def _weighted_mean_pooling(self, hidden, attention_mask):
        """Weighted mean pooling for embeddings."""
        attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
        s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
        d = attention_mask_.sum(dim=1, keepdim=True).float()
        reps = s / d
        return reps
    
    def _encode_image(self, image):
        """Encode images using VisRet."""
        inputs = {
            "text": [''], 
            'image': [image],
            'tokenizer': self.tokenizer
        }
        outputs = self.model(**inputs)
        reps = self._weighted_mean_pooling(outputs.last_hidden_state, outputs.attention_mask)   
        embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
        return embeddings
    
    def _encode_text(self, text):
        """Encode text using VisRet."""
        inputs = {
            "text": [text],
            'image': [None],
            'tokenizer': self.tokenizer
        }
        outputs = self.model(**inputs)
        reps = self._weighted_mean_pooling(outputs.last_hidden_state, outputs.attention_mask)
        embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
        return embeddings
    
    def _process_single_item(self, item_path: str):
        """Process a single image and return embedding."""
        image = Image.open(item_path).convert('RGB')
        embeddings = self._encode_image(image)
        return torch.from_numpy(embeddings[0]), None
    
    
    def _process_single_query(self, query: str):
        """Process a single query and return embedding."""
        INSTRUCTION = "Represent this query for retrieving relevant documents: "
        full_query = INSTRUCTION + query
        embeddings = self._encode_text(full_query)
        return torch.from_numpy(embeddings[0]), None

