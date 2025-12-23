"""
DSE Encoder implementation.
"""

import sys
sys.path.append('/mnt/colpali')

import torch
from PIL import Image, ImageFile
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from encoder import Encode
Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True



class DSEEncoder(Encode):
    """DSE encoder implementation."""
    
    def __init__(self,**kwargs):
        """
        Initialize DSE encoder.
        """
        super().__init__(**kwargs)
    
    def _load_model(self, **kwargs):
        """Load DSE model and processor."""
        min_pixels = 1*28*28
        max_pixels = 2560*28*28
        device = f"cuda:{self.device}"
        
        self.processor = AutoProcessor.from_pretrained(
            "MrLight/dse-qwen2-2b-mrl-v1", 
            min_pixels=min_pixels, 
            max_pixels=max_pixels,
            use_fast=True
        )
        print("processor loaded!")
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            'MrLight/dse-qwen2-2b-mrl-v1', 
            cache_dir="/mnt/transformers-cache",
            dtype=torch.bfloat16,
        ).to(device).eval()
        print("model loaded!")
        
        self.processor.tokenizer.padding_side = "left"
        self.model.padding_side = "left"
    
    def _get_embedding(self, last_hidden_state: torch.Tensor, dimension: int) -> torch.Tensor:
        """Extract embedding from hidden states."""
        return last_hidden_state[:, -1]
    
    def _process_single_item(self, item_path: str):
        """Process a single image and return embedding."""
        doc = Image.open(item_path)
        
        message = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': doc},
                    {'type': 'text', 'text': 'What is shown in this image?'}
                ]
            }
        ]

        doc_texts = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
        ]
        doc_image_inputs, doc_video_inputs = process_vision_info([message])
        doc_inputs = self.processor(
            text=doc_texts, 
            images=doc_image_inputs, 
            videos=doc_video_inputs, 
            padding='longest', 
            return_tensors='pt'
        ).to(f"cuda:{self.device}")
        
        cache_position = torch.arange(0, len(doc_texts))
        doc_inputs = self.model.prepare_inputs_for_generation(**doc_inputs, cache_position=cache_position, use_cache=False)
        
        with torch.no_grad():
            output = self.model(**doc_inputs, return_dict=True, output_hidden_states=True)
        
        doc_embeddings = self._get_embedding(output.hidden_states[-1], 1536)
        doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)
        
        return doc_embeddings, None

    
    def _process_single_query(self, query: str):
        """Process a single query and return embedding."""
        # Create dummy image for text-only processing
        message = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': Image.new('RGB', (28, 28)), 'resized_height':1, 'resized_width':1},
                    {'type': 'text', 'text': f'Query: {query}'},
                ]
            }
        ]

        query_texts = [self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"]
        query_image_inputs, query_video_inputs = process_vision_info(message)
        query_inputs = self.processor(
            text=query_texts, 
            images=query_image_inputs, 
            videos=query_video_inputs, 
            padding='longest', 
            return_tensors='pt'
        ).to(f"cuda:{self.device}")
        
        cache_position = torch.arange(0, len(query_texts))
        query_inputs = self.model.prepare_inputs_for_generation(**query_inputs, cache_position=cache_position, use_cache=False)
        
        with torch.no_grad():
            output = self.model(**query_inputs, return_dict=True, output_hidden_states=True)
        
        query_embeddings = self._get_embedding(output.hidden_states[-1], 1536)

        return query_embeddings, None



