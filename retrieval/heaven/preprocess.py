import warnings
warnings.filterwarnings("ignore")

import torch
import json
import os
import argparse
import nltk
from tqdm import tqdm
from typing import List, Dict, Optional
from transformers import AutoTokenizer


class QueryPreprocessor:
    """Preprocessor for query tokenization and POS tagging."""
    
    # Supported models and their tokenizer paths
    MODEL_TOKENIZERS = {
        'colqwen25': 'Qwen/Qwen2.5-VL-3B-Instruct',
        'colpali': 'google/paligemma-3b-mix-448',
        'colqwen2': 'Qwen/Qwen2-VL-2B-Instruct'
    }
    
    def __init__(self, model: str, hf_token: Optional[str] = None):
        """
        Initialize the query preprocessor.
        
        Args:
            model: Model name (e.g., 'colqwen25', 'colpali')
            hf_token: HuggingFace token for model access
        """
        self.model = model
        self.hf_token = hf_token
        
        # Get tokenizer path
        if model in self.MODEL_TOKENIZERS:
            tokenizer_path = self.MODEL_TOKENIZERS[model]
        else:
            # Assume model name is the tokenizer path
            tokenizer_path = model
        
        # Load tokenizer
        print(f"Loading tokenizer: {tokenizer_path}")
        if hf_token:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=hf_token)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        print(f"Tokenizer loaded for model: {model}")
        
        # Download NLTK data for POS tagging
        self._setup_nltk()
    
    def _setup_nltk(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            nltk.download('maxent_ne_chunker_tab', quiet=True)
            nltk.download('words', quiet=True)
            print("NLTK data downloaded")
    
    def decode_query(self, query_input_ids: torch.Tensor) -> List[Optional[str]]:
        """
        Decode query token IDs to text.
        
        Args:
            query_input_ids: Tensor of token IDs
            
        Returns:
            List of decoded tokens (None for padding/special tokens)
        """
        decoded_tokens = []
        for token_id in query_input_ids:
            if token_id == -1:  # Padding
                decoded_tokens.append(None)
            else:
                decoded_tokens.append(
                    self.tokenizer.decode(token_id, skip_special_tokens=False)
                )
        return decoded_tokens
    
    def clean_tokens(self, tokens: List[Optional[str]]) -> tuple:
        """
        Clean tokenized query by removing special tokens and padding.
        
        Args:
            tokens: List of decoded tokens
            
        Returns:
            Tuple of (mask, cleaned_tokens) where mask indicates valid tokens
        """
        mask = []
        cleaned = []
        
        for token in tokens:
            # Skip None and special tokens
            if token is not None and token != "<|endoftext|>":
                # Normalize uppercase tokens
                if token.upper() == token:
                    token = token.lower()
                cleaned.append(token.strip())
                mask.append(True)
            else:
                mask.append(False)
        
        return mask, cleaned
    
    def get_pos_tags(self, tokens: List[str], mask: List[bool]) -> List[Optional[str]]:
        """
        Get POS tags for tokens using NLTK.
        
        Args:
            tokens: List of cleaned tokens
            mask: Boolean mask indicating valid positions
            
        Returns:
            List of POS tags aligned with original positions
        """
        # Get POS tags from NLTK
        nltk_pos_tags = nltk.pos_tag(tokens)
        
        # Map back to original positions
        pos_tags = []
        nltk_idx = 0
        
        for i, is_valid in enumerate(mask):
            if is_valid:
                pos_tag = nltk_pos_tags[nltk_idx][1]
                # Mark first token as QUERY (special handling)
                if nltk_idx == 0:
                    pos_tag = "QUERY"
                pos_tags.append(pos_tag)
                nltk_idx += 1
            else:
                pos_tags.append(None)
        
        return pos_tags
    
    def tokenize_dataset(
        self,
        folder: str,
        input_file: str = "test.json",
        output_file: str = "test_tokenized.json"
    ) -> Dict:
        """
        Tokenize queries in a dataset.
        
        Args:
            folder: Dataset folder path
            input_file: Input JSON filename
            output_file: Output JSON filename
            
        Returns:
            Tokenized dataset
        """
        dataset_path = f"/data/HEAVEN/benchmark/{folder}"
        output_path = f"{dataset_path}/{output_file}"
        
        # Check if already tokenized
        if os.path.exists(output_path):
            print(f"Loading existing tokenized data from {output_path}")
            with open(output_path, "r") as f:
                return json.load(f)
        
        # Load original data
        input_path = f"{dataset_path}/{input_file}"
        with open(input_path, "r") as f:
            data = json.load(f)
        
        print(f"\nTokenizing {len(data)} queries for {folder}...")
        
        # Load query input IDs
        query_input_ids = self._load_query_input_ids(folder)
        
        # Decode each query
        for i in tqdm(range(len(data)), desc="Decoding"):
            data[i][f'query_tokenized_{self.model}'] = self.decode_query(query_input_ids[i])
        
        # Save tokenized data
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        
        print(f"Tokenized data saved to {output_path}")
        
        return data
    
    def _load_query_input_ids(self, folder: str) -> torch.Tensor:
        """Load query input IDs for a dataset."""
        # Load query input IDs
        path = f"/data/HEAVEN/benchmark/{folder}/query_input_ids_{self.model}.pt"
        
        if os.path.exists(path):
            print(f"Loaded query input IDs from {path}")
            return torch.load(path)
        
        raise FileNotFoundError(f"Query input IDs not found for {folder}")
    
    def add_pos_tags(
        self,
        folder: str,
        input_file: str = "test_tokenized.json",
        output_file: str = "test_tokenized.json"
    ) -> Dict:
        """
        Add POS tags to tokenized dataset.
        
        Args:
            folder: Dataset folder path
            input_file: Input tokenized JSON filename
            output_file: Output JSON filename (can be same as input)
            
        Returns:
            Dataset with POS tags added
        """
        dataset_path = f"/data/HEAVEN/benchmark/{folder}"
        input_path = f"{dataset_path}/{input_file}"
        
        # Load tokenized data
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Tokenized data not found: {input_path}")
        
        with open(input_path, "r") as f:
            data = json.load(f)
        
        # Check if already tagged
        pos_key = f'query_pos_{self.model}'
        if pos_key in data[0]:
            print(f"POS tags already exist for {folder}")
            return data
        
        print(f"\nAdding POS tags for {len(data)} queries in {folder}...")
        
        # Add POS tags to each query
        for i, item in enumerate(tqdm(data, desc="POS Tagging")):
            tokenized_key = f'query_tokenized_{self.model}'
            
            if tokenized_key not in item:
                raise ValueError(f"Query not tokenized. Run tokenize_dataset first.")
            
            # Clean tokens
            mask, cleaned_tokens = self.clean_tokens(item[tokenized_key])
            
            # Get POS tags
            pos_tags = self.get_pos_tags(cleaned_tokens, mask)
            
            # Save POS tags
            data[i][pos_key] = pos_tags
        
        # Save with POS tags
        output_path = f"{dataset_path}/{output_file}"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        
        print(f"POS tags saved to {output_path}")
        
        return data
    
    def preprocess_dataset(
        self,
        folder: str
    ) -> Dict:
        """
        Complete preprocessing: tokenization + POS tagging.
        
        Args:
            folder: Dataset folder path
            
        Returns:
            Fully preprocessed dataset
        """
        print(f"\n{'='*60}")
        print(f"Preprocessing Dataset: {folder}")
        print(f"{'='*60}\n")
        
        # Step 1: Tokenize queries
        data = self.tokenize_dataset(folder)
        
        # Step 2: Add POS tags
        data = self.add_pos_tags(folder)
        
        print(f"\n{'='*60}")
        print(f"Preprocessing Complete for {folder}")
        print(f"{'='*60}\n")
        
        return data
    
    def preprocess_multiple(
        self,
        folders: List[str]
    ):
        """
        Preprocess multiple datasets.
        
        Args:
            folders: List of dataset folder paths
        """
        print(f"\n{'='*60}")
        print(f"Preprocessing {len(folders)} Datasets")
        print(f"{'='*60}\n")
        
        for folder in folders:
            try:
                self.preprocess_dataset(folder)
            except Exception as e:
                print(f"Error processing {folder}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"All Datasets Preprocessed")
        print(f"{'='*60}\n")


def main():
    """Command-line interface for query preprocessing."""
    parser = argparse.ArgumentParser(
        description="Query Preprocessing: Tokenization and POS Tagging"
    )
    
    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        default="colqwen25",
        help="Model name (e.g., 'colqwen25', 'colpali')"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--folder",
        type=str,
        default="ViDoSeek",
        help="Single dataset folder to preprocess"
    )
    
    # Optional arguments
    parser.add_argument(
        "--hf_token",
        type=str,
        help="HuggingFace token for model access"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for loading embeddings (default: 'cpu')"
    )
    
    args = parser.parse_args()

    
    # Initialize preprocessor
    processor = QueryPreprocessor(model=args.model, hf_token=args.hf_token)
    

    # Process each folder
    print(f"\n{'='*60}")
    print(f"Query Filtering: {args.folder} with model {args.model}")
    print(f"{'='*60}\n")
    
    # Complete preprocessing (tokenization + POS tagging)
    processor.preprocess_dataset(args.folder)
    
    print(f"\n{'='*60}")
    print("Query Filtering Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

