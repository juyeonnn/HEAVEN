import warnings
warnings.filterwarnings("ignore")

import torch
import json
import os
from tqdm import tqdm
import argparse
import numpy as np
import torch.nn.functional as F

from utils import *


class BaselineRetrieval:
    """Baseline retrieval system for document/page-level retrieval."""
    
    # Datasets that use document-level by default
    DOC_LEVEL_DATASETS = ['M3DocVQA']
    
    def __init__(
        self,
        folder: str,
        model: str,
        device: str = "0",
        doc_level: bool = False
    ):
        """
        Initialize the baseline retrieval system.
        
        Args:
            folder: Dataset folder name
            model: Model name for embeddings
            device: CUDA device ID
            doc_level: Whether to use document-level retrieval
        """
        self.folder = folder
        self.model = model
        self.device = f"cuda:{device}"
        
        # Set doc_level to True for M3DocVQA by default
        if folder in self.DOC_LEVEL_DATASETS:
            self.doc_level = True
        else:
            self.doc_level = doc_level
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load dataset, embeddings, and mappings."""
        # Determine split key
        split_key = get_split_key(self.folder) if self.doc_level else None
        
        # Load test data
        test_path = f"/data/HEAVEN/benchmark/{self.folder}/test.json"
        with open(test_path, "r") as f:
            data = json.load(f)
        
        # Prepare ground truth document names
        if self.folder == 'M3DocVQA':
            self.gt_doc_names = prepare_data(data, split_key=None)
        else:
            self.gt_doc_names = prepare_data(data, split_key=split_key)
        
        print(f"Loaded {len(data)} samples from {self.folder}")
        
        # Load embedding mappings
        mapping_path = f"/data/HEAVEN/benchmark/{self.folder}/embeddings_mapping.json"
        with open(mapping_path, "r") as f:
            embedding_mapping = json.load(f)
        self.doc_mapping = get_doc_mapping(embedding_mapping, split_key=split_key)
        
        # Load embeddings
        self.image_embeddings = load_embedding(
            f"/data/HEAVEN/benchmark/{self.folder}/embeddings_{self.model}.pt",
            self.device
        )
        
        # Load query embeddings
        query_path = f"/data/HEAVEN/benchmark/{self.folder}/query_embeddings_{self.model}.pt"
        self.query_embeddings = load_embedding(query_path, self.device)
    
    def compute_scores(self, batch_size: int = 128) -> torch.Tensor:
        """
        Compute similarity scores between queries and images.
        
        Args:
            batch_size: Batch size for multi-vector scoring
            
        Returns:
            Similarity scores tensor
        """
        # Check if model uses multi-vector scoring
        is_multi_vector = (
            self.model.startswith('col') or 
            self.model == 'bge_m3_multi'
        )
        
        if is_multi_vector:
            scores = score_multi_vector(
                self.query_embeddings,
                self.image_embeddings,
                device=self.device,
                batch_size=batch_size
            )
        else:
            scores = torch.matmul(
                self.query_embeddings.to(self.device),
                self.image_embeddings.transpose(-2, -1).to(self.device)
            )
        
        return scores
    
    def aggregate_scores(self, scores: torch.Tensor) -> tuple:
        """
        Aggregate page-level scores to document-level scores.
        
        Args:
            scores: Page-level similarity scores
            
        Returns:
            Tuple of (doc_scores, gt_doc_ids)
        """
        doc_score_mapping = {}
        doc_scores = []
        
        for mapping_idx, (doc, idx) in enumerate(self.doc_mapping.items()):
            doc_score_mapping[doc] = mapping_idx
            max_val, _ = scores[:, idx].max(dim=1)
            doc_scores.append(max_val)
        
        doc_scores = torch.stack(doc_scores, dim=0).T
        gt_doc_ids = [[doc_score_mapping[i] for i in item] for item in self.gt_doc_names]
        
        return doc_scores, gt_doc_ids
    

    def run(self, batch_size: int = 128) -> dict:
        """
        Run the complete retrieval pipeline.
        
        Args:
            batch_size: Batch size for scoring
            
        Returns:
            Evaluation results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Running Baseline Retrieval")
        print(f"{'='*60}")
        print(f"Folder: {self.folder}")
        print(f"Model: {self.model}")
        print(f"Doc-level: {self.doc_level}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Compute scores
        scores = self.compute_scores(batch_size=batch_size)
        
        # Aggregate to document level
        doc_scores, gt_doc_ids = self.aggregate_scores(scores)
        
        # Evaluate
        results = evaluate(doc_scores, gt_doc_ids)

        return results


def main():
    """Command-line interface for baseline retrieval."""
    parser = argparse.ArgumentParser(
        description="Baseline Retrieval System for Document Retrieval"
    )
    
    # Required arguments
    parser.add_argument(
        "--folder",
        type=str,
        default='ViMDoc',
        help="Dataset folder name (e.g., 'ViMDoc', 'OpenDocVQA', 'M3DocVQA')"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name for embeddings (e.g., 'dse', 'gme2b', 'colpali', 'bge_m3_multi')"
    )
    
    # Optional arguments
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device ID (default: '0')"
    )
    parser.add_argument(
        "--doc_level",
        action="store_true",
        help="Use document-level retrieval (automatically enabled for M3DocVQA)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for multi-vector scoring (default: 256)"
    )
    
    args = parser.parse_args()
    
    # Initialize and run retrieval
    retrieval = BaselineRetrieval(
        folder=args.folder,
        model=args.model,
        device=args.device,
        doc_level=args.doc_level
    )
    
    results = retrieval.run(batch_size=args.batch_size)
    
    print(f"\n{'='*60}")
    print("Retrieval Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
