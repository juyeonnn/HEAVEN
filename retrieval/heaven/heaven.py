import warnings
warnings.filterwarnings("ignore")

import torch
import json
import os
import argparse
import numpy as np

from utils import *
from preprocess import QueryPreprocessor
from stage1 import Stage1Retrieval
from stage2 import Stage2Retrieval


class HEAVEN:
    """
    HEAVEN: Hybrid-Vector Retrieval combining Stage 1 (VS-Page Filtering) and Stage 2 (Query Token Filtering).
    """
    
    def __init__(
        self,
        folder: str,
        stage1_model: str,
        stage2_model: str,
        device: str = "0",
        doc_level: bool = False,
        # Stage 1 parameters
        reduction_factor: int = 15,
        layout_fname: str = "layout",
        alpha: float = 0.1,
        filter_ratio_stage1: float = 0.5,
        # Stage 2 parameters
        k: int = 200,
        filter_ratio_stage2: float = 0.25,
        beta: float = 0.3,
        batch_size: int = 256
    ):
        """
        Initialize HEAVEN retrieval.
        
        Args:
            folder: Dataset folder name
            stage1_model: Single-vector model for stage 1 (e.g., 'dse')
            stage2_model: Multi-vector model for stage 2 (e.g., 'colqwen25')
            device: CUDA device ID
            doc_level: Whether to use document-level retrieval
            reduction_factor: Reduction factor for vs-page creation
            layout_fname: Layout filename
            alpha: Weight for vs-page scores in stage 1
            filter_ratio_stage1: Filter ratio for stage 1
            k: Top-K for filtering in stage 2
            filter_ratio_stage2: Filter ratio for refinement in stage 2
            beta: Weight for stage 1 scores in final combination
            batch_size: Batch size for multi-vector scoring (default: 256)
        """
        self.folder = folder
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.device = device
        
        print(f"\n{'='*60}")
        print("Initializing HEAVEN Retrieval")
        print(f"{'='*60}")
        print(f"Dataset: {folder}")
        print(f"Stage 1 Model: {stage1_model}")
        print(f"Stage 2 Model: {stage2_model}")
        print(f"Device: cuda:{device}")
        print(f"\nStage 1 Parameters:")
        print(f"  Reduction Factor: {reduction_factor}")
        print(f"  Alpha: {alpha}")
        print(f"  Filter Ratio: {filter_ratio_stage1}")
        print(f"\nStage 2 Parameters:")
        print(f"  K: {k}")
        print(f"  Filter Ratio: {filter_ratio_stage2} (K Refine: {int(k * filter_ratio_stage2)})")
        print(f"  Beta: {beta}")
        print(f"  Batch Size: {batch_size}")
        print(f"{'='*60}\n")
        
        # Initialize Stage 1 retrieval
        self.stage1 = Stage1Retrieval(
            folder=folder,
            model=stage1_model,
            device=device,
            doc_level=doc_level,
            reduction_factor=reduction_factor,
            layout_fname=layout_fname,
            alpha=alpha,
            filter_ratio=filter_ratio_stage1
        )
        
        # Initialize Stage 2 retrieval
        self.stage2 = Stage2Retrieval(
            folder=folder,
            model=stage2_model,
            stage1_model=stage1_model,
            device=device,
            doc_level=doc_level,
            k=k,
            filter_ratio=filter_ratio_stage2,
            beta=beta,
            batch_size=batch_size
        )
    
    def preprocess_if_needed(self) -> bool:
        """
        Check and run preprocessing if needed (tokenization + POS tagging).
        
        Returns:
            True if preprocessing was run, False if already existed
        """
        tokenized_path = f"/data/HEAVEN/benchmark/{self.folder}/test_tokenized.json"
        
        if not os.path.exists(tokenized_path):
            print(f"Need to preprocess {self.folder}")
            preprocessor = QueryPreprocessor(model=self.stage2_model)
            preprocessor.preprocess_dataset(self.folder)
            return True
        
        # Check if POS tags exist
        with open(tokenized_path, "r") as f:
            data = json.load(f)
        if f'query_pos_{self.stage2_model}' not in data[0]:
            print(f"Need to add POS tags for {self.folder}")
            preprocessor = QueryPreprocessor(model=self.stage2_model)
            preprocessor.add_pos_tags(self.folder)
            return True
        
        print("Query preprocessing already completed")
        return False
    
    def run(self) -> dict:
        """Run the complete HEAVEN pipeline."""
        print(f"\n{'='*60}")
        print("Running HEAVEN Pipeline")
        print(f"{'='*60}\n")
        
        # Stage 1: VS-Page filtering
        print(f"\n{'='*60}")
        print("Stage 1: VS-Page Filtering")
        print(f"{'='*60}\n")
        
        vs_page_scores, page_scores = self.stage1.compute_scores()
        stage1_scores, _ = self.stage1.filter_and_combine(vs_page_scores, page_scores)
        
        # Convert to tensor on CPU for stage 2
        stage1_scores = stage1_scores.cpu()
        
        # Stage 2: Query token Filtering
        print(f"\n{'='*60}")
        print("Stage 2: Query Token Filtering")
        print(f"{'='*60}\n")
        
        # Save stage 1 scores for stage 2 to use
        os.makedirs("score_cache", exist_ok=True)
        stage1_key = f"_rf{self.stage1.reduction_factor}_alpha{self.stage1.alpha}_filter{self.stage1.filter_ratio}"
        stage1_score_path = f"score_cache/stage1_{self.folder}_{self.stage1_model}{stage1_key}.pt"
        torch.save(stage1_scores, stage1_score_path)
        print(f"Saved stage 1 scores to {stage1_score_path}")
        
        # Update stage2 to use this stage1_key
        self.stage2.stage1_key = stage1_key
        self.stage2._load_or_compute_scores()
        
        # Run stage 2 filtering and combination
        final_scores = self.stage2.filter_and_combine()
        
        # Prepare for evaluation (use stage 2's mapping and gt)
        doc_scores, gt_doc_ids = prepare_evaluate(
            final_scores, 
            self.stage2.page_mapping, 
            self.stage2.gt_doc_names
        )
        
        # Evaluate
        print(f"\n{'='*60}")
        print("Final Evaluation")
        print(f"{'='*60}")
        results = evaluate(doc_scores, gt_doc_ids)

        print(f"\n{'='*60}")
        print("HEAVEN Retrieval Complete!")
        print(f"{'='*60}\n")
        
        return results


def main():
    """Command-line interface for HEAVEN."""
    parser = argparse.ArgumentParser(
        description="HEAVEN: Hybrid-Vector Retrieval for Visually Rich Documents"
    )
    
    # Required arguments
    parser.add_argument(
        "--folder",
        type=str,
        default='ViMDoc',
        help="Dataset folder name (e.g., 'ViMDoc', 'OpenDocVQA', 'M3DocVQA')"
    )
    parser.add_argument(
        "--stage1_model",
        type=str,
        default='dse',
        help="Single-vector model for stage 1 (e.g., 'dse', 'gme2b')"
    )
    parser.add_argument(
        "--stage2_model",
        type=str,
        default='colqwen25',
        help="Multi-vector model for stage 2 (e.g., 'colqwen25')"
    )
    
    # Stage 1 parameters
    parser.add_argument(
        "--reduction_factor",
        type=int,
        default=15,
    help="Reduction factor for vs-page construction (default: 15)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Stage 1: Weight for vs-page scores (default: 0.1)"
    )
    parser.add_argument(
        "--filter_ratio_stage1",
        type=float,
        default=0.5,
        help="Stage 1: Filter ratio for page filtering (default: 0.5)"
    )
    
    # Stage 2 parameters
    parser.add_argument(
        "--k",
        type=int,
        default=200,
        help="Stage 2: Top-K for stage 1 scores (default: 200)"
    )
    parser.add_argument(
        "--filter_ratio_stage2",
        type=float,
        default=0.25,
        help="Stage 2: Filter ratio for refinement (default: 0.25)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Stage 2: Weight for stage 1 score (default: 0.3)"
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
        help="CUDA device ID (default: '0')"
    )
    parser.add_argument(
        "--doc_level",
        action="store_true",
        help="Use document-level retrieval (auto-enabled for M3DocVQA)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for multi-vector scoring (default: 256)"
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run preprocessing (tokenization + POS tagging) if needed"
    )
    
    args = parser.parse_args()
    
    # Initialize HEAVEN
    heaven = HEAVEN(
        folder=args.folder,
        stage1_model=args.stage1_model,
        stage2_model=args.stage2_model,
        device=args.device,
        doc_level=args.doc_level,
        reduction_factor=args.reduction_factor,
        layout_fname=args.layout_fname,
        alpha=args.alpha,
        filter_ratio_stage1=args.filter_ratio_stage1,
        k=args.k,
        filter_ratio_stage2=args.filter_ratio_stage2,
        beta=args.beta,
        batch_size=args.batch_size
    )
    
    # Run preprocessing if query preprocessor is needed
    if args.preprocess:
        heaven.preprocess_if_needed()
    
    # Run retrieval
    results = heaven.run()
    
    return results


if __name__ == "__main__":
    main()
