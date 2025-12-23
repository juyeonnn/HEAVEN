import warnings
warnings.filterwarnings("ignore")

import torch
import json
import os
import argparse
import numpy as np
import torch.nn.functional as F

from utils import *
from colpali_engine.compression.token_pooling import HierarchicalTokenPooler
from colpali_engine.models import ColQwen2_5_Processor


class PoolingRetrieval:

    # Datasets that use document-level by default
    DOC_LEVEL_DATASETS = ['M3DocVQA']
    
    def __init__(
        self,
        folder: str,
        model: str,
        device: str = "0",
        doc_level: bool = False,
        pooling_type: str = "document",
        pool_factor: int = 2,
        special_token_num: int = 10
    ):
        """
        Initialize pooling-based retrieval.
        
        Args:
            folder: Dataset folder name
            model: Model name for embeddings
            device: CUDA device ID
            doc_level: Whether to use document-level retrieval
            pooling_type: Type of pooling ("document" or "query")
            pool_factor: Pooling factor (reduces tokens by this factor)
            special_token_num: Number of special query tokens to pool (for query pooling)
        """
        self.folder = folder
        self.model = model
        self.device = f"cuda:{device}"
        self.pooling_type = pooling_type
        self.pool_factor = pool_factor
        self.special_token_num = special_token_num
        
        # Set doc_level for M3DocVQA
        if folder in self.DOC_LEVEL_DATASETS:
            self.doc_level = True
        else:
            self.doc_level = doc_level
        
        # Validate multi-vector model
        self.is_multi_vector = model.startswith('col')
        if not self.is_multi_vector:
            raise ValueError(f"Pooling only supports multi-vector models, got: {model}")
        
        # Initialize token pooler
        self._init_pooler()
        
        # Load data
        self._load_data()
        
        # Apply pooling
        self._apply_pooling()
    
    def _init_pooler(self):
        """Initialize the hierarchical token pooler."""
        self.processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")
        self.token_pooler = HierarchicalTokenPooler()
        print("Initialized HierarchicalTokenPooler")
    
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
    
    def _apply_pooling(self):
        """Apply pooling to embeddings based on pooling type."""
        if self.pooling_type == "document":
            self._pool_document_patches()
        elif self.pooling_type == "query":
            self._pool_query_tokens()
        else:
            raise ValueError(f"Invalid pooling_type: {self.pooling_type}")
    
    def _pool_document_patches(self):
        """
        Pool document patches using hierarchical token pooling.
        Reduces the number of patches by pool_factor.
        """
        original_shape = self.image_embeddings.shape
        
        print(f"\nDocument Patch Pooling:")
        print(f"  Original shape: {original_shape}")
        print(f"  Pool factor: {self.pool_factor}")
        
        self.image_embeddings = self.token_pooler.pool_embeddings(
            self.image_embeddings,
            pool_factor=self.pool_factor,
            padding=True,
            padding_side=self.processor.tokenizer.padding_side,
            num_workers=32
        )
        
        print(f"  Pooled shape: {self.image_embeddings.shape}")
        print(f"  Reduction: {original_shape[1]} → {self.image_embeddings.shape[1]} patches\n")
    
    def _pool_query_tokens(self):
        """
        Pool query tokens using hierarchical token pooling.
        First selects last special_token_num tokens, then pools them by pool_factor.
        """
        original_shape = self.query_embeddings.shape
        
        # Select last special_token_num tokens
        query_subset = self.query_embeddings[:, -self.special_token_num:, :]
        
        print(f"\nQuery Token Pooling:")
        print(f"  Original shape: {original_shape}")
        print(f"  Selected last {self.special_token_num} special tokens: {query_subset.shape}")
        print(f"  Pool factor: {self.pool_factor}")
        
        self.query_embeddings = self.token_pooler.pool_embeddings(
            query_subset,
            pool_factor=self.pool_factor,
            padding_side="left",
            padding=True
        )
        
        print(f"  Pooled shape: {self.query_embeddings.shape}")
        print(f"  Reduction: {self.special_token_num} → {self.query_embeddings.shape[1]} tokens\n")
    
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
    
    def save_results(self, results: dict):
        """Save retrieval results to JSON file."""
        os.makedirs("results_pooling", exist_ok=True)
        
        if self.doc_level:
            file_path = f"results_pooling/{self.folder}_doc.json"
        else:
            file_path = f"results_pooling/{self.folder}.json"
        
        # Load existing results
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                all_results = json.load(f)
        else:
            all_results = {}
        
        # Create model name with pooling info
        if self.pooling_type == "document":
            model_name = f"{self.model}_doc_pool_{self.pool_factor}"
        else:
            model_name = f"{self.model}_query_pool_{self.pool_factor}_tok{self.special_token_num}"
        
        # Add pooling configuration
        results['pooling_type'] = self.pooling_type
        results['pool_factor'] = self.pool_factor
        if self.pooling_type == "document":
            results['embedding_shape'] = list(self.image_embeddings.shape)
        else:
            results['special_token_num'] = self.special_token_num
            results['query_shape'] = list(self.query_embeddings.shape)
        
        all_results[model_name] = results
        
        with open(file_path, "w") as f:
            json.dump(all_results, f, indent=4)
        
        print(f"Results saved to {file_path}")
    
    def run(self, batch_size: int = 128) -> dict:
        """Run the complete pooling-based retrieval pipeline."""
        print(f"\n{'='*60}")
        print(f"Pooling-Based Retrieval")
        print(f"{'='*60}")
        print(f"Folder: {self.folder}")
        print(f"Model: {self.model}")
        print(f"Pooling Type: {self.pooling_type}")
        print(f"Pool Factor: {self.pool_factor}")
        if self.pooling_type == "query":
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
        
        # Save results
        self.save_results(results)
        
        return results


def main():
    """Command-line interface for pooling-based retrieval."""
    parser = argparse.ArgumentParser(
        description="Pooling-Based Retrieval for Efficiency"
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
        help="Multi-vector model name (e.g., 'colpali', 'colqwen2.5')"
    )
    
    # Pooling arguments
    parser.add_argument(
        "--pooling_type",
        type=str,
        default="document",
        choices=["document", "query"],
        help="Type of pooling: 'document' for patch pooling, 'query' for token pooling"
    )
    parser.add_argument(
        "--pool_factor",
        type=int,
        default=2,
        help="Pooling factor (reduces tokens by this factor, default: 2)"
    )
    parser.add_argument(
        "--special_token_num",
        type=int,
        default=10,
        help="Number of special tokens to pool for query pooling (default: 10)"
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
    
    # Initialize and run pooling retrieval
    retrieval = PoolingRetrieval(
        folder=args.folder,
        model=args.model,
        device=args.device,
        doc_level=args.doc_level,
        pooling_type=args.pooling_type,
        pool_factor=args.pool_factor,
        special_token_num=args.special_token_num
    )
    
    results = retrieval.run(batch_size=args.batch_size)
    
    print(f"\n{'='*60}")
    print("Pooling Retrieval Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
