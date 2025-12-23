import warnings
warnings.filterwarnings("ignore")

import torch
import json
import os
import argparse
import numpy as np
import torch.nn.functional as F

from utils import *


class PruningRetrieval:
    
    # Datasets that use document-level by default
    DOC_LEVEL_DATASETS = ['M3DocVQA']
    
    def __init__(
        self,
        folder: str,
        model: str,
        device: str = "0",
        doc_level: bool = False,
        pruning_type: str = "document",
        pruning_ratio: float = 0.5,
        special_token_num: int = 10
    ):
        """
        Initialize pruning-based retrieval.
        
        Args:
            folder: Dataset folder name
            model: Model name for embeddings
            device: CUDA device ID
            doc_level: Whether to use document-level retrieval
            pruning_type: Type of pruning ("document" or "query")
            pruning_ratio: Ratio of patches/tokens to keep (for document pruning)
            special_token_num: Number of special tokens to keep (for query pruning)
        """
        self.folder = folder
        self.model = model
        self.device = f"cuda:{device}"
        self.pruning_type = pruning_type
        self.pruning_ratio = pruning_ratio
        self.special_token_num = special_token_num
        
        # Set doc_level for M3DocVQA
        if folder in self.DOC_LEVEL_DATASETS:
            self.doc_level = True
        else:
            self.doc_level = doc_level
        
        # Validate multi-vector model
        self.is_multi_vector = model.startswith('col') or model == 'bge_m3_multi'
        if not self.is_multi_vector:
            raise ValueError(f"Pruning only supports multi-vector models, got: {model}")
        
        # Load data
        self._load_data()
        
        # Apply pruning
        self._apply_pruning()
    
    def _load_data(self):
        """Load dataset, embeddings, and mappings."""
        split_key = get_split_key(self.folder) if self.doc_level else None
        
        # Load test data
        with open(f"/data/HEAVEN/benchmark/{self.folder}/test.json", "r") as f:
            data = json.load(f)
        
        if self.folder == 'M3DocVQA':
            self.gt_doc_names = prepare_data(data, split_key=None)
        else:
            self.gt_doc_names = prepare_data(data, split_key=split_key)
        
        print(f"Loaded {len(data)} samples from {self.folder}")
        
        # Load mappings
        with open(f"/data/HEAVEN/benchmark/{self.folder}/embeddings_mapping.json", "r") as f:
            embedding_mapping = json.load(f)
        self.doc_mapping = get_doc_mapping(embedding_mapping, split_key=split_key)
        
        # Load embeddings
        self.image_embeddings = load_embedding(
            f"/data/HEAVEN/benchmark/{self.folder}/embeddings_{self.model}.pt",
            self.device
        )
        self.query_embeddings = load_embedding(
            f"/data/HEAVEN/benchmark/{self.folder}/query_embeddings_{self.model}.pt",
            self.device
        )
    
    def _apply_pruning(self):
        """Apply pruning to embeddings based on pruning type."""
        if self.pruning_type == "document":
            self._prune_document_patches()
        elif self.pruning_type == "query":
            self._prune_query_tokens()
        else:
            raise ValueError(f"Invalid pruning_type: {self.pruning_type}")
    
    def _prune_document_patches(self):
        """
        Prune document patches randomly while keeping special tokens.
        Keeps the last 7 tokens which are typically special tokens in ColPali.
        """
        num_image, num_patch, embed_dim = self.image_embeddings.shape
        
        # Number of regular patches (excluding special tokens)
        num_regular_patches = num_patch - 7
        num_keep = int(num_regular_patches * self.pruning_ratio)
        
        print(f"\nDocument Patch Pruning:")
        print(f"  Original patches: {num_patch} ({num_regular_patches} regular + 7 special)")
        print(f"  Keeping: {num_keep} regular patches ({self.pruning_ratio:.1%})")
        print(f"  Final patches: {num_keep + 7}")
        
        # Random pruning for each image
        pruned_embeddings = torch.zeros((num_image, num_keep, embed_dim), device=self.device)
        
        for i in range(num_image):
            # Random indices from regular patches
            idx = torch.randperm(num_regular_patches)[:num_keep]
            pruned_embeddings[i] = self.image_embeddings[i, idx]
        
        # Concatenate with special tokens (last 7 tokens)
        self.image_embeddings = torch.cat([
            pruned_embeddings,
            self.image_embeddings[:, -7:, :]
        ], dim=1)
        
        print(f"  Pruned embeddings shape: {self.image_embeddings.shape}\n")
    
    def _prune_query_tokens(self):
        """
        Prune query tokens by keeping only the last N special tokens.
        """
        original_shape = self.query_embeddings.shape
        self.query_embeddings = self.query_embeddings[:, -self.special_token_num:, :]
        
        print(f"\nQuery Token Pruning:")
        print(f"  Original tokens: {original_shape[1]}")
        print(f"  Keeping last: {self.special_token_num} special tokens")
        print(f"  Pruned shape: {self.query_embeddings.shape}\n")
    
    def compute_scores(self, batch_size: int = 128) -> torch.Tensor:
        """
        Compute similarity scores using multi-vector scoring.
        
        Args:
            batch_size: Batch size for scoring
            
        Returns:
            Similarity scores tensor
        """
        scores = score_multi_vector(
            self.query_embeddings,
            self.image_embeddings,
            device=self.device,
            batch_size=batch_size
        )
        return scores
    
    def aggregate_scores(self, scores: torch.Tensor) -> tuple:
        """Aggregate page-level scores to document-level scores."""
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
        """Run the complete pruning-based retrieval pipeline."""
        print(f"\n{'='*60}")
        print(f"Pruning-Based Retrieval")
        print(f"{'='*60}")
        print(f"Folder: {self.folder}")
        print(f"Model: {self.model}")
        print(f"Pruning Type: {self.pruning_type}")
        if self.pruning_type == "document":
            print(f"Pruning Ratio: {self.pruning_ratio}")
        else:
            print(f"Special Token Num: {self.special_token_num}")
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
    """Command-line interface for pruning-based retrieval."""
    parser = argparse.ArgumentParser(
        description="Pruning-Based Retrieval for Efficiency"
    )
    
    # Required arguments
    parser.add_argument(
        "--folder",
        type=str,
        default='ViMDoc',
        help="Dataset folder name"
    )
    parser.add_argument(
        "--model",
        type=str,
        default='colqwen25',
        help="Multi-vector model name (e.g., 'colpali', 'colqwen2.5', 'bge_m3_multi')"
    )
    
    # Pruning arguments
    parser.add_argument(
        "--pruning_type",
        type=str,
        default="document",
        choices=["document", "query"],
        help="Type of pruning: 'document' for patch pruning, 'query' for token pruning"
    )
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        default=0.5,
        help="Ratio of patches to keep for document pruning (default: 0.5)"
    )
    parser.add_argument(
        "--special_token_num",
        type=int,
        default=10,
        help="Number of special tokens to keep for query pruning (default: 10)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device ID"
    )
    parser.add_argument(
        "--doc_level",
        action="store_true",
        help="Use document-level retrieval"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for scoring"
    )
    
    args = parser.parse_args()
    
    # Initialize and run pruning retrieval
    retrieval = PruningRetrieval(
        folder=args.folder,
        model=args.model,
        device=args.device,
        doc_level=args.doc_level,
        pruning_type=args.pruning_type,
        pruning_ratio=args.pruning_ratio,
        special_token_num=args.special_token_num
    )
    
    results = retrieval.run(batch_size=args.batch_size)
    
    print(f"\n{'='*60}")
    print("Pruning Retrieval Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
