"""
Stage 2 Retrieval: Multi-Vector Reranking of Pages (Query Token Filtering)
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import json
import os
import argparse
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from utils import *


class Stage2Retrieval:
    # Datasets that use document-level by default
    DOC_LEVEL_DATASETS = ['M3DocVQA']
    
    def __init__(
        self,
        folder: str,
        model: str,
        stage1_model: str,
        device: str = "0",
        doc_level: bool = False,
        k: int = 200,
        filter_ratio: float = 0.25,
        beta: float = 0.3,
        stage1_key: str = "",
        batch_size: int = 256
    ):
        """
        Args:
            folder: Dataset folder name
            model: Multi-vector model name for token filtering
            stage1_model: Single-vector retrieval model for stage 1
            device: CUDA device ID
            doc_level: Whether to use document-level retrieval
            k: Top-K for filtering with key tokens
            filter_ratio: Ratio for further filtering, where k_refine = k * filter_ratio
            beta: Weight for stage 1 scores
            stage1_key: Suffix for stage 1 score file
        """
        self.folder = folder
        self.model = model
        self.stage1_model = stage1_model
        self.device = f"cuda:{device}"
        self.k = k
        self.filter_ratio = filter_ratio
        self.k_refine = int(k * filter_ratio)
        self.beta = beta
        self.stage1_key = stage1_key
        self.batch_size = batch_size
        # Set doc_level for M3DocVQA
        if folder in self.DOC_LEVEL_DATASETS:
            self.doc_level = True
        else:
            self.doc_level = doc_level
        
        # Load data
        self._load_data()
        
        # Load or compute scores
        self._load_or_compute_scores()
    
    def _load_data(self):
        """Load dataset and mappings."""
        split_key = get_split_key(self.folder) if self.doc_level else None
        
        # Load tokenized test data
        with open(f"/data/HEAVEN/benchmark/{self.folder}/test_tokenized.json", "r") as f:
            self.data = json.load(f)
        
        if self.folder == 'M3DocVQA':
            self.gt_doc_names = prepare_data(self.data, split_key=None)
        else:
            self.gt_doc_names = prepare_data(self.data, split_key=split_key)
        
        print(f"Loaded {len(self.data)} samples from {self.folder}")
        
        # Load page mapping
        with open(f"/data/HEAVEN/benchmark/{self.folder}/embeddings_mapping.json", "r") as f:
            embedding_mapping = json.load(f)
        self.page_mapping = get_doc_mapping(embedding_mapping, split_key=split_key)
        
        # Load embeddings
        self.image_embeddings = load_embedding(
            f"/data/HEAVEN/benchmark/{self.folder}/embeddings_{self.model}.pt",
            self.device
        )
        self.query_embeddings = load_embedding(
            f"/data/HEAVEN/benchmark/{self.folder}/query_embeddings_{self.model}.pt",
            self.device
        )
        
        print(f"Image embeddings: {self.image_embeddings.shape}")
        print(f"Query embeddings: {self.query_embeddings.shape}")
    
    def _load_or_compute_scores(self):
        """Load or compute the three types of scores."""
        os.makedirs("score_cache", exist_ok=True)
        
        # Decompose query embeddings into key tokens (nouns) and non-key tokens (others)
        self._decompose_query_embeddings()
        
        # Load or compute stage 1 (single-vector baseline) scores
        self._load_or_compute_stage1_scores()
        
        # Load or compute key token (noun) scores
        self._load_or_compute_key_scores()
        
        # Load or compute non-key token (other) scores
        self._load_or_compute_non_key_scores()
    
    def _decompose_query_embeddings(self):
        """Decompose query embeddings based on POS tags (nouns vs others)."""
        score_path1 = f"score_cache/score_key_{self.folder}_{self.model}.pt"
        score_path2 = f"score_cache/score_non_key_{self.folder}_{self.model}.pt"
        
        if os.path.exists(score_path1) and os.path.exists(score_path2):
            print("Scores already computed, skipping query filtering")
            return
        
        print("Decomposing query embeddings by POS tags...")
        
        query_key_list = []
        query_non_key_list = []
        
        for query_idx, item in enumerate(self.data):
            key_indices, non_key_indices = split_query(item[f'query_pos_{self.model}'])
            query_key_list.append(self.query_embeddings[query_idx, key_indices])
            query_non_key_list.append(self.query_embeddings[query_idx, non_key_indices])
        
        self.query_key = pad_sequence(query_key_list, batch_first=True, padding_value=0)
        self.query_non_key = pad_sequence(query_non_key_list, batch_first=True, padding_value=0)
        
        print(f"Query key tokens (nouns): {self.query_key.shape}")
        print(f"Query non-key tokens (others): {self.query_non_key.shape}")
        
        # Save token numbers
        num_path = f'/data/HEAVEN/benchmark/{self.folder}/num_token.json'
        if os.path.exists(num_path):
            with open(num_path, 'r') as f:
                num_token = json.load(f)
        else:
            num_token = {}

        num_token[args.model] = [self.query_key.shape[1], self.query_non_key.shape[1]]

        with open(num_path, 'w') as f:
            json.dump(num_token, f, indent=4)
        print(f"Token numbers saved: {num_token[args.model]}")
    
    def _load_or_compute_stage1_scores(self):
        """Load or compute stage 1 (single-vector baseline) scores."""
        os.makedirs("score_cache", exist_ok=True)
        score_path = f"score_cache/stage1_{self.folder}_{self.stage1_model}{self.stage1_key}.pt"
        
        if os.path.exists(score_path):
            self.score_stage1 = load_embedding(score_path, device='cpu')
            
            ret_q = load_embedding(
                f"/data/HEAVEN/benchmark/{self.folder}/query_embeddings_{self.stage1_model}.pt",
                self.device
            )
            self.embedding_dim = ret_q.shape[-1]
            del ret_q
            
            print(f"Loaded stage 1 scores from {score_path}")
        else:
            print("Computing stage 1 scores...")
            torch.cuda.empty_cache()
            
            ret_query_embeddings = load_embedding(
                f"/data/HEAVEN/benchmark/{self.folder}/query_embeddings_{self.stage1_model}.pt",
                self.device
            )
            ret_image_embeddings = load_embedding(
                f"/data/HEAVEN/benchmark/{self.folder}/embeddings_{self.stage1_model}.pt",
                self.device
            )
            
            self.embedding_dim = ret_query_embeddings.shape[-1]
            
            self.score_stage1 = torch.matmul(
                ret_query_embeddings,
                ret_image_embeddings.transpose(-2, -1)
            ).cpu()
            
            torch.save(self.score_stage1, score_path)
            print(f"Stage 1 scores saved to {score_path}")
            
            del ret_query_embeddings, ret_image_embeddings
            torch.cuda.empty_cache()
    
    def _load_or_compute_key_scores(self):
        """Load or compute key token (noun) scores."""
        score_path = f"score_cache/stage2_score_key_{self.folder}_{self.model}.pt"
        
        if os.path.exists(score_path):
            self.score_key = load_embedding(score_path, device='cpu')
            print(f"Loaded key token scores from {score_path}")
        else:
            print("Computing key token (noun) scores...")
            torch.cuda.empty_cache()
            
            self.score_key = score_multi_vector(
                self.query_key,
                self.image_embeddings,
                device=self.device,
                batch_size=self.batch_size
            )
            
            torch.save(self.score_key, score_path)
            print(f"Key token scores saved to {score_path}")
    
    def _load_or_compute_non_key_scores(self):
        """Load or compute non-key token (other) scores."""
        score_path = f"score_cache/score_non_key_{self.folder}_{self.model}.pt"
        
        if os.path.exists(score_path):
            self.score_non_key = load_embedding(score_path, device='cpu')
            print(f"Loaded non-key token scores from {score_path}")
        else:
            print("Computing non-key token (other) scores...")
            torch.cuda.empty_cache()
            
            self.score_non_key = score_multi_vector(
                self.query_non_key,
                self.image_embeddings,
                device=self.device,
                batch_size=self.batch_size
            )
            
            torch.save(self.score_non_key, score_path)
            print(f"Non-key token scores saved to {score_path}")
            
            # Clean up if we computed all scores
            if hasattr(self, 'query_key'):
                del self.query_key, self.query_non_key, self.image_embeddings
                torch.cuda.empty_cache()
    
    def filter_and_combine(self) -> torch.Tensor:
        """
        Apply hierarchical filtering: Stage 1 → Key Tokens → Non-Key Tokens.
        
        Returns:
            Combined final scores
        """
        print(f"\n{'='*60}")
        print("Applying Hierarchical Filtering")
        print(f"{'='*60}")
        
        # Step 1: Filter top-K using stage 1 scores
        print(f"Step 1: Filtering top-{self.k} using stage 1 scores")
        top_k_scores_l1, top_k_indices_l1 = self.score_stage1.topk(k=self.k, dim=1)
        
        # Create key token mask
        key_masks = torch.zeros_like(self.score_stage1, dtype=torch.bool)
        key_masks.scatter_(1, top_k_indices_l1, True)
        
        # Apply mask to key token scores
        self.score_key = self.score_key * key_masks.float()
        
        print(f"  Kept {self.k} out of {self.score_stage1.shape[1]} pages")
        print(f"  Mask ratio: {key_masks.sum().item()/key_masks.numel()*100:.2f}%")
        
        # Step 2: Further refine using key token scores
        print(f"Step 2: Refining to top-{self.k_refine} from {self.k} using key token scores (filter_ratio={self.filter_ratio})")
        masked_score_key = torch.where(
            key_masks,
            self.score_key,
            torch.tensor(-float('inf'))
        )
        top_k_scores_l2, top_k_indices_l2 = masked_score_key.topk(k=self.k_refine, dim=1)
        
        # Create non-key token mask
        non_key_masks = torch.zeros_like(self.score_key, dtype=torch.bool)
        non_key_masks.scatter_(1, top_k_indices_l2, True)
        
        # Apply mask to non-key token scores
        self.score_non_key = self.score_non_key * non_key_masks.float()
        
        print(f"  Kept {self.k_refine} out of {self.k} pages")
        print(f"  Mask ratio: {non_key_masks.sum().item()/non_key_masks.numel()*100:.2f}%")
        print(f"{'='*60}\n")
        
        # Combine all scores
        final_scores = self.beta * self.score_stage1 + (1 - self.beta) * (self.score_key + self.score_non_key)
        
        return final_scores
    
    def run(self) -> dict:
        """Run the complete two-stage retrieval pipeline with query token filtering."""
        print(f"\n{'='*60}")
        print(f"Stage 2 Retrieval (Query Token Decomposition)")
        print(f"{'='*60}")
        print(f"Folder: {self.folder}")
        print(f"Model: {self.model}")
        print(f"Retrieval Model: {self.stage1_model}")
        print(f"Doc-level: {self.doc_level}")
        print(f"K: {self.k}")
        print(f"Filter Ratio: {self.filter_ratio} (K Refine: {self.k_refine})")
        print(f"Beta: {self.beta}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Apply hierarchical filtering
        final_scores = self.filter_and_combine()
        
        # Aggregate to document level
        doc_scores, gt_doc_ids = prepare_evaluate(final_scores, self.page_mapping, self.gt_doc_names)
        
        # Evaluate
        results = evaluate(doc_scores, gt_doc_ids)
        
        return results


def main():
    """Command-line interface for stage 2 retrieval."""
    parser = argparse.ArgumentParser(
        description="Stage 2: Two-Stage Retrieval with Query Token Decomposition (Key vs Non-Key Tokens)"
    )
    
    # Required arguments
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Dataset folder name"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Multi-vector model for stage 2 (e.g., 'colqwen25')"
    )
    parser.add_argument(
        "--stage1_model",
        type=str,
        required=True,
        help="Single-vector model for stage 1 (e.g., 'dse')"
    )
    
    # Two-stage arguments
    parser.add_argument(
        "--k",
        type=int,
        default=200,
        help="Top-K for stage 1 scores (default: 200)"
    )
    parser.add_argument(
        "--filter_ratio",
        type=float,
        default=0.25,
        help="Filter ratio for stage 2, where k_refine = k * filter_ratio (default: 0.25)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Weight for stage 1 score (default: 0.3)"
    )
    parser.add_argument(
        "--stage1_key",
        type=str,
        default="_rf15_alpha0.1_filter0.5",
        help="Suffix for stage 1 score file (default: '_rf15_alpha0.1_filter0.5')"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for multi-vector scoring (default: 256)"
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
    
    args = parser.parse_args()
    
    # Initialize and run stage 2 retrieval
    retrieval = Stage2Retrieval(
        folder=args.folder,
        model=args.model,
        stage1_model=args.stage1_model,
        device=args.device,
        doc_level=args.doc_level,
        k=args.k,
        filter_ratio=args.filter_ratio,
        beta=args.beta,
        stage1_key=args.stage1_key,
        batch_size=args.batch_size
    )
    
    results = retrieval.run()
    
    print(f"\n{'='*60}")
    print("Stage 2 Retrieval Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
