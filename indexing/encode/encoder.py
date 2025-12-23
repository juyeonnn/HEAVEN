import os
import json
import torch
import torch.nn.functional as F
import argparse
from abc import ABC, abstractmethod
import importlib
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Encode(ABC):

    def __init__(self, device: str = "0", **kwargs):
        """
        Initialize the encoder.
        
        Args:
            device: CUDA device to use
            dataset: Dataset name
            **kwargs: Additional model-specific arguments
        """
        self.device = device
        self.model = None
        self.processor = None
        
        # Initialize model and processor
        self._load_model(**kwargs)
    
    @abstractmethod
    def _load_model(self, **kwargs):
        """Load the specific model and processor. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _process_single_item(self, item_path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process a single item (image or text) and return embeddings and optional input_ids.
        
        Args:
            item_path: Path to the item to process
            
        Returns:
            Tuple of (embeddings, input_ids) where input_ids can be None
        """
        pass
    @abstractmethod
    def _process_single_query(self, query: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process a single query and return embeddings and optional input_ids.
        
        Args:
            query: Query text to process
            
        Returns:
            Tuple of (embeddings, input_ids) where input_ids can be None
        """
        pass
    
    def filter_files(self, files: List[str]) -> List[str]:
        """Filter files to only include supported image formats."""
        return [f for f in files if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
    
    def clean_name(self, name: str) -> str:
        """Clean filename by removing extensions."""
        return name.replace('.png','').replace('.jpg','').replace('.jpeg','')
    
    def create_mapping(self, files: List[str]) -> Dict[str, int]:
        """Create mapping dictionary from filenames to indices."""
        return {self.clean_name(files[i]): i for i in range(len(files))}
    
    def save_mapping(self, mapping_dict: Dict[str, int]):
        """Save mapping dictionary to JSON file."""
        with open(self.mapping_path, "w") as f:
            json.dump(mapping_dict, f)
        print(f"Saved mapping_dict to {self.mapping_path}")
    
    def process_dataset(self, encoder_type: str, dataset: str, save_embeddings: bool = True, save_input_ids: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process the entire dataset and return embeddings and input_ids.
        
        Args:
            encoder_type: Type of encoder
            save_embeddings: Whether to save embeddings to file
            save_input_ids: Whether to save input_ids to file
            
        Returns:
            Tuple of (embeddings, input_ids)
        """


        save_input_ids = True if encoder_type in ['colpali', 'colqwen2', 'colqwen25'] else False
        
        self.dataset_path = f'/mnt/HEAVEN/benchmark/{dataset}/pages'
        # self.dataset_path = f'/workspace/HEAVEN-main/benchmark/{dataset}/sampled_pages'
        self.mapping_path = f"/mnt/HEAVEN/benchmark/{dataset}/embeddings_mapping.json"


        # Get list of files
        files = sorted(os.listdir(self.dataset_path))
        files = self.filter_files(files)
        
        # Create and save mapping
        mapping_dict = self.create_mapping(files)
        self.save_mapping(mapping_dict)
        
        # Convert to full paths
        file_paths = [os.path.join(self.dataset_path, f) for f in files]
        
        # Process all items
        embeddings = []
        input_ids = []
        
        for file_path in tqdm(file_paths, desc="Processing items", ncols=100, total=len(file_paths)):
            
            embedding, input_id = self._process_single_item(file_path)
            embeddings.append(embedding.squeeze(0) if embedding.dim() > 1 else embedding)
            if save_input_ids:
                input_ids.append(input_id.squeeze(0) if input_id.dim() > 1 else input_id)
            
            # Clear GPU memory after each item
            torch.cuda.empty_cache()
        
        # Pad sequences if needed
        if embedding.dim() == 3:
            embeddings = pad_sequence(embeddings, batch_first=True, padding_side='left', padding_value=0)
            if save_input_ids:
                input_ids = pad_sequence(input_ids, batch_first=True, padding_side='left', padding_value=-1)
        else:
            embeddings = torch.stack(embeddings)
        

        # Save results
        if save_embeddings:
            self._save_embeddings(embeddings, dataset, encoder_type)
        if save_input_ids:
            self._save_input_ids(input_ids, dataset, encoder_type)
        
        return embeddings, input_ids

    
    
    def process_queries(self, encoder_type: str, dataset: str, id_col: str = 'id', query_col: str = 'query', 
                        save_embeddings: bool = True, save_input_ids: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process queries from a dataset.
        
        Args:
            dataset: Dataset name
            id_col: Column name for query IDs
            query_col: Column name for query text
            save_embeddings: Whether to save embeddings to file
            save_input_ids: Whether to save input_ids to file
            
        Returns:
            Tuple of (embeddings, input_ids)
        """

        save_input_ids = True if encoder_type in ['colpali', 'colqwen2', 'colqwen25'] else False
        
        # with open(f"/mnt/HEAVEN/benchmark/{dataset}/test.json", "r") as f:
        with open(f"/workspace/HEAVEN-main/benchmark/{dataset}/test.json", "r") as f:
            query_data = json.load(f)
        
        all_queries = [ex[query_col] for ex in query_data]
        all_ids = [ex[id_col] for ex in query_data]
        
        # Create query embedding mapping
        query_embedding_mapping = {all_ids[i]: i for i in range(len(all_ids))}
        
        # Process queries
        all_query_embeddings = []
        all_input_ids = []
        
        for query in tqdm(all_queries, ncols=100, desc="Processing queries", total=len(all_queries)):
            embedding, input_id = self._process_single_query(query)
            all_query_embeddings.append(embedding.squeeze(0) if embedding.dim() > 1 else embedding)
            if input_id is not None:
                all_input_ids.append(input_id.squeeze(0) if input_id.dim() > 1 else input_id)
        
        # Pad sequences if needed
        if embedding.dim() == 3:
            all_query_embeddings = pad_sequence(all_query_embeddings, batch_first=True, padding_value=0, padding_side='left')
            if save_input_ids:
                all_input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=-1, padding_side='left')
        else:
            all_query_embeddings = torch.stack(all_query_embeddings)
        
        # Save results
        if save_embeddings:
            self._save_query_embeddings(all_query_embeddings, dataset,  encoder_type)
        if save_input_ids:
            self._save_query_input_ids(all_input_ids, dataset, encoder_type)
        
        return all_query_embeddings, all_input_ids


    def process_vs_page(self, encoder_type: str, dataset: str, save_embeddings: bool = True, save_input_ids: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process vs-page from a dataset.
        
        Args:
            dataset: Dataset name
            save_embeddings: Whether to save embeddings to file
            save_input_ids: Whether to save input_ids to file
        """
        self.dataset_path = f"/mnt/HEAVEN/benchmark/{dataset}/vs-page"
        self.mapping_path = f"/mnt/HEAVEN/benchmark/{dataset}/vs-page_embeddings_mapping.json"
        
        # Get list of files
        files = sorted(os.listdir(self.dataset_path))
        files = self.filter_files(files)
        print(f"Found {len(files)} vs-page images")
        
        # Create and save mapping
        mapping_dict = self.create_mapping(files)
        self.save_mapping(mapping_dict)
        
        # Convert to full paths
        file_paths = [os.path.join(self.dataset_path, f) for f in files]
        print(f"Processing {len(file_paths)} vs-page images")
        
        # Process all items
        embeddings = []
        
        for file_path in tqdm(file_paths, desc="Processing vs-page", ncols=100, total=len(file_paths)):
            embedding, _ = self._process_single_item(file_path)
            embeddings.append(embedding.squeeze(0) if embedding.dim() > 1 else embedding)
        
            # Clear GPU memory after each item
            torch.cuda.empty_cache()
        
    
        embeddings = torch.stack(embeddings)
        # Save results
        if save_embeddings:
            self._save_vs_page_embeddings(embeddings, dataset, encoder_type)

        return embeddings

    
    def _save_vs_page_embeddings(self, embeddings: torch.Tensor, dataset: str, encoder_type: str):
        """Save vs-page embeddings to file. Must be implemented by subclasses."""
        emb_path = f"/mnt/HEAVEN/benchmark/{dataset}/vs-page_embeddings_{encoder_type}.pt"
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        torch.save(embeddings, emb_path)
        print(f"Saved vs-page embeddings {embeddings.shape} to {emb_path}")
    

    def _save_embeddings(self, embeddings: torch.Tensor, dataset: str, encoder_type: str):
        """Save embeddings to file. Must be implemented by subclasses."""
        emb_path = f"/mnt/HEAVEN/benchmark/{dataset}/embeddings_{encoder_type}.pt"
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        torch.save(embeddings, emb_path)
        print(f"Saved embeddings {embeddings.shape} to {emb_path}")
    
    def _save_input_ids(self, input_ids: torch.Tensor, dataset: str, encoder_type: str):
        """Save input_ids to file. Must be implemented by subclasses."""
        input_ids_path = f"/mnt/HEAVEN/benchmark/{dataset}/input_ids_{encoder_type}.pt"
        torch.save(input_ids, input_ids_path)
        print(f"Saved input_ids {input_ids.shape} to {input_ids_path}")
    

    def _save_query_embeddings(self, embeddings: torch.Tensor, dataset: str, encoder_type: str):
        """Save query embeddings to file. Must be implemented by subclasses."""
        emb_path = f"/mnt/HEAVEN/benchmark/{dataset}/query_embeddings_{encoder_type}.pt"
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        torch.save(embeddings, emb_path)
        print(f"Saved query embeddings {embeddings.shape} to {emb_path}")
    

    def _save_query_input_ids(self, input_ids: torch.Tensor, dataset: str, encoder_type: str):
        """Save query input_ids to file. Must be implemented by subclasses."""
        input_ids_path = f"/mnt/HEAVEN/benchmark/{dataset}/query_input_ids_{encoder_type}.pt"
        torch.save(input_ids, input_ids_path)
        print(f"Saved query input_ids {input_ids.shape} to {input_ids_path}")
    
    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        torch.cuda.empty_cache()


import importlib
from typing import Type

def get_encoder_class(encoder_type: str) -> Type:
    """
    Retrieve an encoder class based on the encoder type.

    Args:
        encoder_type (str): Type of encoder. Supported types:
            - colpali
            - colqwen2
            - colqwen25
            - gme
            - dse
            - visret
            - bge_m3
            - bge_m3_multi
            - nvembedv2

    Returns:
        Type: The encoder class corresponding to the given encoder_type.

    Raises:
        ValueError: If encoder_type is not recognized.
        ImportError: If the corresponding encoder module cannot be imported.
        AttributeError: If the class is missing in the module.
    """
    encoder_map = {
        'colpali': ('colpali_encoder', 'ColPaliEncoder'),
        'colqwen2': ('colqwen2_encoder', 'ColQwen2Encoder'),
        'colqwen25': ('colqwen25_encoder', 'ColQwen25Encoder'),
        'gme': ('gme_encoder', 'GMEEncoder'),
        'dse': ('dse_encoder', 'DSEEncoder'),
        'visret': ('visret_encoder', 'VisRetEncoder'),
        'bge_m3': ('bge_m3_encoder', 'BGEM3Encoder'),
        'bge_m3_multi': ('bge_m3_multi_encoder', 'BGEM3MultiEncoder'),
        'nvembedv2': ('nvembedv2_encoder', 'NVEmbedV2Encoder'),
    }

    if encoder_type not in encoder_map:
        valid_types = ', '.join(encoder_map.keys())
        raise ValueError(f"Unknown encoder type '{encoder_type}'. Available types: {valid_types}")

    module_name, class_name = encoder_map[encoder_type]

    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_name}': {e}")
    except AttributeError:
        raise AttributeError(f"'{class_name}' not found in module '{module_name}'.")

def create_encoder(encoder_type: str, **kwargs):
    """
    Create an encoder instance.
    
    Args:
        encoder_type: Type of encoder
        **kwargs: Arguments to pass to encoder
        
    Returns:
        Encoder instance
    """
    encoder_class = get_encoder_class(encoder_type)
    return encoder_class(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Encoder")
    parser.add_argument("--encoder_type", type=str, required=True, 
                       choices=['colpali', 'colqwen2', 'colqwen25', 'gme', 'dse', 'visret', 'bge_m3', 'bge_m3_multi', 'nvembedv2'],
                       help="Type of encoder to use")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    parser.add_argument("--dataset", type=str, default="ViMDoc", help="Dataset folder",
                        choices=['ViMDoc','OpenDocVQA','ViDoSeek','M3DocVQA'])
    args = parser.parse_args()
    
    # Create encoder and process dataset
    encoder = None
    emb_path = f"/mnt/HEAVEN/benchmark/{args.dataset}/embeddings_{args.encoder_type}.pt"

    encoder = create_encoder(args.encoder_type)
    encoder.process_dataset(encoder_type=args.encoder_type, dataset=args.dataset)
    encoder.process_queries(encoder_type=args.encoder_type, dataset=args.dataset)

