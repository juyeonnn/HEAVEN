"""
NV-Embed-v2 Encoder implementation.
"""

import torch
import torch.nn.functional as F
import json
import os
import math
from transformers import AutoModel
from encoder import Encode
from tqdm import tqdm


class NVEmbedV2Encoder(Encode):
    """NV-Embed-v2 encoder implementation."""
    
    def __init__(self, batch_size: int = 32, max_length: int = 8192, **kwargs):
        """
        Initialize NV-Embed-v2 encoder.
        
        Args:
            device: CUDA device to use
            batch_size: Batch size for processing
            **kwargs: Additional arguments
        """
        self.batch_size = batch_size
        self.max_length = max_length
        super().__init__(**kwargs)
    
    def _load_model(self, **kwargs):
        """Load NV-Embed-v2 model."""
        self.model = AutoModel.from_pretrained(
            'nvidia/NV-Embed-v2',
            cache_dir="/mnt/transformers-cache",
            device_map=f'cuda:{self.device}',
            trust_remote_code=True,
        ).eval()
    
    def _load_ocr_data(self):
        """Load OCR data from JSON file."""
        ocr_mapping_path = f'/mnt/HEAVEN/benchmark/{self.dataset}/ocr.json'
        print(f"Loading OCR mapping from {ocr_mapping_path}")
        with open(ocr_mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _process_single_item(self, item_path: str):
        """NV-Embed-v2 processes text, not images directly."""
        pass
    
    def _process_single_query(self, query: str):
        """Process a single query and return embedding."""
        pass
    
    def process_dataset(self, dataset: str, save_embeddings: bool = True, save_input_ids: bool = True):
        """Process OCR text data with NV-Embed-v2."""
        self.dataset = dataset
        self.mapping_path = f"/mnt/HEAVEN/benchmark/{dataset}/embeddings_mapping.json"
        # Load OCR data
        ocr_data = self._load_ocr_data()
        
        # Extract text data and create mapping
        texts = []
        mapping_dict = {}
        
        for i, item in enumerate(ocr_data):
            image_name = item['image']
            text_content = item['text']
            
            # Clean image name (remove extension)
            clean_name = self.clean_name(image_name)
            
            texts.append(text_content)
            mapping_dict[clean_name] = i
        
        # Save mapping dictionary
        self.save_mapping(mapping_dict)
        
        print(f"Processing {len(texts)} texts in batches of {self.batch_size}")
        
        # Process texts in batches
        passage_prefix = ""
        embeddings_all = []
        
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, len(texts), self.batch_size), total=num_batches, desc="Processing texts", ncols=100):
            batch_texts = texts[i:i + self.batch_size]
            
            with torch.no_grad():
                batch_embedding = self.model.encode(
                    batch_texts, 
                    instruction=passage_prefix, 
                    max_length=self.max_length
                )
                
                embeddings_all.append(batch_embedding)
        
        
        # Concatenate all embeddings
        embeddings_all = torch.cat(embeddings_all, dim=0)
        # Normalize embeddings
        embeddings_all = F.normalize(embeddings_all, p=2, dim=1)
        
        if save_embeddings:
            self._save_embeddings(embeddings_all)
        
        return embeddings_all, None
    
    def process_queries(self, dataset: str, id_col: str = 'id', query_col: str = 'query', 
                        save_embeddings: bool = True, save_input_ids: bool = False):
        """Process queries with NV-Embed-v2."""
        self.dataset = dataset
        with open(f"/mnt/HEAVEN/benchmark/{dataset}/test.json", "r") as f:
            query_data = json.load(f)
        
        print("Processing queries...")
        
        # Extract queries and IDs from data
        all_queries = [ex[query_col] for ex in query_data]
        all_ids = [ex[id_col] for ex in query_data]

        
        
        # Create query embedding mapping
        query_embedding_mapping = {all_ids[i]: i for i in range(len(all_ids))}
        
        # Save mapping
        mapping_path = f"/mnt/HEAVEN/benchmark/{dataset}/query_embeddings_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(query_embedding_mapping, f)
        print(f"Saved query mapping to {mapping_path}")
        
        print(f"Processing {len(all_queries)} queries in batches of {self.batch_size}")
        
        # Process queries in batches
        query_prefix = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
        embeddings_all = []

        num_batches = (len(all_queries) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, len(all_queries), self.batch_size), total=num_batches, desc="Processing queries", ncols=100):
            batch_queries = [query_prefix + query for query in all_queries[i:i + self.batch_size]]
            with torch.no_grad():
                batch_embedding = self.model.encode(batch_queries, max_length=self.max_length)
            embeddings_all.append(batch_embedding)
        
        # Concatenate all embeddings
        embeddings_all = torch.cat(embeddings_all, dim=0)

        if save_embeddings:
            self._save_query_embeddings(embeddings_all)
        
        return embeddings_all, None


