"""
Stage 1 Retrieval: Single-Vector Retrieval of Candidate Pages (VS-Page Filtering)
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import json
import os
import argparse
import numpy as np
import torch.nn.functional as F

from utils import *


class Stage1Retrieval:

    # Datasets that use document-level by default
    DOC_LEVEL_DATASETS = ['M3DocVQA']
    
    def __init__(
        self,
        folder: str,
        model: str,
        device: str = "0",
        doc_level: bool = False,
        reduction_factor: int = 15,
        layout_fname: str = "layout",
        alpha: float = 0.1,
        filter_ratio: float = 0.5
    ):
        """
        Initialize two-stage retrieval.
        
        Args:
            folder: Dataset folder name
            model: Model name for embeddings
            device: CUDA device ID
            doc_level: Whether to use document-level retrieval
            reduction_factor: Reduction factor used for vs-page creation
            layout_fname: Layout filename
            alpha: Weight for vs-page scores (1-alpha for page scores)
            filter_ratio: Ratio of pages to keep after filtering
        """
        self.folder = folder
        self.model = model
        self.device = f"cuda:{device}"
        self.reduction_factor = reduction_factor
        self.layout_fname = layout_fname
        self.alpha = alpha
        self.filter_ratio = filter_ratio
        
        # Set doc_level for M3DocVQA
        if folder in self.DOC_LEVEL_DATASETS:
            self.doc_level = True
        else:
            self.doc_level = doc_level
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load dataset, embeddings, and mappings."""
        # For page-level: split_key=None
        # For doc-level: split_key=get_split_key()
        split_key = get_split_key(self.folder) if self.doc_level else None
        
        # Load test data
        with open(f"/data/HEAVEN/benchmark/{self.folder}/test.json", "r") as f:
            data = json.load(f)
        
        if self.folder == 'M3DocVQA':
            self.gt_doc_names = prepare_data(data, split_key=None)
        else:
            self.gt_doc_names = prepare_data(data, split_key=split_key)
        
        print(f"Loaded {len(data)} samples from {self.folder}")
        
        # Load page mapping
        with open(f"/data/HEAVEN/benchmark/{self.folder}/embeddings_mapping.json", "r") as f:
            embedding_mapping = json.load(f)
        self.page_mapping = get_doc_mapping(embedding_mapping, split_key=split_key)
        
        # Load chunk mapping (vs-page)
        chunk_mapping_path = f"/data/HEAVEN/benchmark/{self.folder}/vs-page_embeddings_mapping.json"
        with open(chunk_mapping_path, "r") as f:
            embedding_mapping = json.load(f)
        self.chunk_mapping = get_doc_mapping(embedding_mapping, split_key=None)
        
        # Load chunk to page mapping
        chunk2page_path = f"/data/HEAVEN/benchmark/{self.folder}/{self.layout_fname}_chunk_mapping_postprocessed.json"
        with open(chunk2page_path, "r") as f:
            chunk2page_mapping = json.load(f)
        
        # Create page to chunk index mapping
        self.page2chunkidx_mapping = {}
        for chunk_name, page_names in chunk2page_mapping.items():
            for page_name in page_names:
                if page_name not in self.page2chunkidx_mapping:
                    self.page2chunkidx_mapping[page_name] = []
                self.page2chunkidx_mapping[page_name].extend(self.chunk_mapping[chunk_name])
        
        # Remove duplicates
        for page_name in self.page2chunkidx_mapping:
            self.page2chunkidx_mapping[page_name] = list(set(self.page2chunkidx_mapping[page_name]))
        
        # Load embeddings
        self.vs_page_embeddings = load_embedding(
            f"/data/HEAVEN/benchmark/{self.folder}/vs-page_embeddings_{self.model}.pt",
            self.device
        )
        self.page_embeddings = load_embedding(
            f"/data/HEAVEN/benchmark/{self.folder}/embeddings_{self.model}.pt",
            self.device
        )
        self.query_embeddings = load_embedding(
            f"/data/HEAVEN/benchmark/{self.folder}/query_embeddings_{self.model}.pt",
            self.device
        )
        
        print(f"VS-Page embeddings: {self.vs_page_embeddings.shape}")
        print(f"Page embeddings: {self.page_embeddings.shape}")
        print(f"Query embeddings: {self.query_embeddings.shape}")
    
    def compute_scores(self) -> tuple:
        """
        Compute scores for two-stage retrieval.
        
        Returns:
            Tuple of (vs_page_scores, page_scores)
        """
        # Stage 1: VS-Page scores
        vs_page_scores = torch.matmul(
            self.query_embeddings,
            self.vs_page_embeddings.transpose(-2, -1)
        )
        
        # Stage 2: Page scores
        page_scores = torch.matmul(
            self.query_embeddings,
            self.page_embeddings.transpose(-2, -1)
        )
        
        print(f"VS-Page scores: {vs_page_scores.shape}")
        print(f"Page scores: {page_scores.shape}")
        
        return vs_page_scores, page_scores
    
    def filter_and_combine(
        self,
        vs_page_scores: torch.Tensor,
        page_scores: torch.Tensor
    ) -> tuple:
        """
        Filter pages based on vs-page scores and combine with page scores.
        
        Args:
            vs_page_scores: VS-page similarity scores
            page_scores: Page similarity scores
            
        Returns:
            Tuple of (final_scores, gt_doc_ids)
        """
        # Map vs-page scores to page level
        page_vs_scores = []
        
        for page, _ in self.page_mapping.items():
            chunk_idx = self.page2chunkidx_mapping[page]
            if len(chunk_idx) > 1:
                page_vs_scores.append(vs_page_scores[:, chunk_idx].max(dim=1)[0])
            else:
                page_vs_scores.append(vs_page_scores[:, chunk_idx].squeeze(1))
        
        page_vs_scores = torch.stack(page_vs_scores, dim=0).T
        
        # Apply filtering
        if self.filter_ratio != 1.0:
            topk = int(self.filter_ratio * page_scores.shape[1])
            print(f"Filtering to top {topk} pages (ratio: {self.filter_ratio})")
            mask, _ = filter_doc(page_vs_scores, topk)
            page_scores = page_scores * mask.float()
        
        # Combine scores
        final_scores = self.alpha * page_vs_scores + (1 - self.alpha) * page_scores
        
        # Prepare for evaluation
        doc_scores, gt_doc_ids = prepare_evaluate(final_scores, self.page_mapping, self.gt_doc_names)
        
        return doc_scores, gt_doc_ids
    
    
    def run(self) -> dict:
        """Run the complete two-stage retrieval pipeline."""
        print(f"\n{'='*60}")
        print(f"Stage 1 Two-Stage Retrieval")
        print(f"{'='*60}")
        print(f"Folder: {self.folder}")
        print(f"Model: {self.model}")
        print(f"Doc-level: {self.doc_level}")
        print(f"Reduction Factor: {self.reduction_factor}")
        print(f"Alpha: {self.alpha}")
        print(f"Filter Ratio: {self.filter_ratio}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Compute scores
        vs_page_scores, page_scores = self.compute_scores()
        
        # Filter and combine
        doc_scores, gt_doc_ids = self.filter_and_combine(vs_page_scores, page_scores)
        
        # Evaluate
        results = evaluate(doc_scores, gt_doc_ids)
        
        return results


def main():
    """Command-line interface for stage 1 retrieval."""
    parser = argparse.ArgumentParser(
        description="Stage 1: Two-Stage Retrieval with VS-Page Filtering"
    )
    
    # Required arguments
    parser.add_argument(
        "--folder",
        type=str,
        default="ViDoSeek",
        help="Dataset folder name"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gme",
        help="Model name for embeddings"
    )
    
    # Two-stage arguments
    parser.add_argument(
        "--reduction_factor",
        type=int,
        default=15,
        help="Reduction factor used for vs-page creation (default: 15)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Weight for vs-page scores, 1-alpha for page scores (default: 0.1)"
    )
    parser.add_argument(
        "--filter_ratio",
        type=float,
        default=0.5,
        help="Ratio of pages to keep after filtering (default: 0.5)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--layout_fname",
        type=str,
        default="layout",
        help="Layout filename (default: 'layout')"
    )
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
    
    # Initialize and run stage 1 retrieval
    retrieval = Stage1Retrieval(
        folder=args.folder,
        model=args.model,
        device=args.device,
        doc_level=args.doc_level,
        reduction_factor=args.reduction_factor,
        layout_fname=args.layout_fname,
        alpha=args.alpha,
        filter_ratio=args.filter_ratio
    )
    
    results = retrieval.run()
    
    print(f"\n{'='*60}")
    print("Stage 1 Retrieval Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
