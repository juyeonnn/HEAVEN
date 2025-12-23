"""
BGE-M3 Encoder implementation.
"""

import torch
import json
import os
from typing import List, Dict
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from encoder import Encode


class BGEM3Encoder(Encode):
    """BGE-M3 encoder implementation."""
    
    def __init__(self, batch_size: int = 16, max_length: int = 8192, **kwargs):
        """
        Initialize BGE-M3 encoder.
        
        Args:
            device: CUDA device to use
            batch_size: Batch size for processing
            **kwargs: Additional arguments
        """
        self.batch_size = batch_size
        self.max_length = max_length
        super().__init__(**kwargs)
    
    def _load_model(self, **kwargs):
        """Load BGE-M3 model."""
        self.model = BGEM3FlagModel(
            'BAAI/bge-m3',
            use_fp16=True,
            device=f'cuda:{self.device}'
        )
    
    def _load_ocr_data(self):
        """Load OCR data from JSON file."""
        ocr_mapping_path = f'/mnt/HEAVEN/benchmark/{self.dataset}/ocr.json'
        print(f"Loading OCR mapping from {ocr_mapping_path}")
        with open(ocr_mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _process_single_item(self, item_path: str):
        pass

    def _process_single_query(self, query: str):
        pass

    def process_dataset(self, dataset: str, save_embeddings: bool = True, save_input_ids: bool = False):
        """Process OCR text data with BGE-M3."""
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
        embeddings_all = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            with torch.no_grad():
                batch_result = self.model.encode(
                    batch_texts, 
                    batch_size=len(batch_texts),
                    max_length=self.max_length,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
                batch_embedding = torch.tensor(batch_result['dense_vecs'])
                embeddings_all.append(batch_embedding)
        
        # Concatenate all embeddings
        embeddings_all = torch.cat(embeddings_all, dim=0)
        
        if save_embeddings:
            self._save_embeddings(embeddings_all)
        
        return embeddings_all, None
    
    
    
    
    def process_queries(self, dataset: str, id_col: str = 'id', query_col: str = 'query', 
                        save_embeddings: bool = True, save_input_ids: bool = False):
        """Process queries and return embeddings."""
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
        embeddings_all = []
        
        for i in range(0, len(all_queries), self.batch_size):
            batch_queries = all_queries[i:i + self.batch_size]
            
            with torch.no_grad():
                batch_result = self.model.encode(
                    batch_queries, 
                    batch_size=len(batch_queries),
                    max_length=self.max_length,  
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                    verbose=False
                )
                batch_embedding = torch.tensor(batch_result['dense_vecs'])
                embeddings_all.append(batch_embedding)
        
        # Concatenate all embeddings
        embeddings_all = torch.cat(embeddings_all, dim=0)
        
        if save_embeddings:
            self._save_query_embeddings(embeddings_all)
        
        return embeddings_all, None

