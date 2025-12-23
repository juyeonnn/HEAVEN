# HEAVEN: Hybrid-Vector Retrieval for Visually Rich Documents

<a href='https://arxiv.org/abs/2510.22215'><img src='https://img.shields.io/badge/arXiv-2510.22215-b31b1b.svg'></a>
<a href='https://huggingface.co/datasets/kaistdata/ViMDoc'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-green'></a>
<a href='https://opensource.org/licenses/MIT'><img src='https://img.shields.io/badge/License-MIT-yellow.svg'></a>

Official Repository for our paper **"Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy"**



## 🔥News

- **ViMDoc** is now available on [Hugging Face🤗](https://huggingface.co/datasets/kaistdata/ViMDoc)!

## 0. ViMDoc Benchmark 
**ViMDoc** (Visually-rich Long Multi-Document Retrieval Benchmark) for evaluating visual document
retrieval under both **multi-document** and **long-document** settings. 


```python
from datasets import load_dataset

dataset = load_dataset("kaistdata/ViMDoc", split="ViMDoc")
```

### Format

The sample dataset contains `sample_query.json` with queries and ground truth document IDs:
```json
{
    "id": "<query_id>",
    "query": "<query_text>",
    "doc_ids": [
        "<document_id>"
    ]
}
```
Sample document pages are stored in `sample_pages/`.

**Note:** Full datasets for other benchmarks are available from their original sources: [OpenDocVQA](https://huggingface.co/datasets/NTT-hil-insight/OpenDocVQA) | [ViDoSeek](https://huggingface.co/datasets/autumncc/ViDoSeek) | [M3DocVQA](https://github.com/bloomberg/m3docrag)


## 1. Indexing
### 1.1 Encoding (Query/Document)

```bash
cd indexing/encode

# Visusal encoder
python encoder.py --encoder_type dse --folder ViMDoc
python encoder.py --encoder_type colqwen25 --folder ViMDoc

# Textual encoder
python ocr.py --device 0 --folder ViMDoc
python encoder.py --encoder_type nvembedv2 --folder ViMDoc
python encoder.py --encoder_type bge_m3_multi --folder ViMDoc
```

### Available Encoders

| Encoder | Modality | Type | HF Checkpoint |
|---------|----------|------|------------|
| `colpali` | Visusal | Multi-Vector | `vidore/colpali-v1.3` |
| `colqwen2` | Visusal | Multi-Vector | `vidore/colqwen2-v1.0` |
| `colqwen25` | Visusal | Multi-Vector | `vidore/colqwen2.5-v0.2` |
| `gme` | Visusal | Single-Vector | `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct` |
| `dse` | Visusal | Single-Vector | `MrLight/dse-qwen2-2b-mrl-v1` |
| `visret` | Visusal | Single-Vector | `openbmb/VisRAG-Ret` |
| `bge_m3_multi` | Textual (OCR) | Multi-Vector | `BAAI/bge-m3` |
| `bge_m3` | Textual (OCR) | Single-Vector | `BAAI/bge-m3` |
| `nvembedv2` | Textual (OCR) | Single-Vector | `nvidia/NV-Embed-v2` |


### 1.2 VS-Page Construction

```bash

cd indexing/vs-page

# Step 1: Document Layout Analysis
python DLA.py --dataset ViMDoc --device 0

# Step 2: Assemble & VS-page Encoding
python assemble.py \
    --dataset ViMDoc \
    --encoder_type dse \
    --reduction_factor 15 \
    --device 0
```


## 2. Retrieval - HEAVEN

Run the complete HEAVEN pipeline (Stage 1 + Stage 2):

```bash
cd retrieval/heaven

python heaven.py \
    --folder ViMDoc \
    --stage1_model dse \
    --stage2_model colqwen25 \
    --device 0 \
    --preprocess
```

**Stage 1 Only** :
```bash
python stage1.py --folder ViMDoc --model dse --alpha 0.1 --filter_ratio 0.5
```

**Stage 2 Only** :
```bash
# Preprocess queries first
python preprocess.py --folder ViMDoc --model colqwen25

# Run Stage 2
python stage2.py --folder ViMDoc --model colqwen25 --stage1_model dse --k 200 --filter_ratio 0.25
```

## Structure
```
HEAVEN/
│
├── benchmark/                    
│   ├── ViMDoc/                  
│   ├── OpenDocVQA/            
│   ├── ViDoSeek/                
│   └── M3DocVQA/
│       
├── indexing/                      
│   ├── encode/                  
│   └── vs-page/
│               
├── retrieval/                    
│   ├── baeline/                   
│   └── heaven/
│                
└── run.sh              
```

## Citation

```bibtex
@article{kim2025hybrid,
  title={Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy},
  author={Kim, Juyeon and Lee, Geon and Choi, Dongwon and Kim, Taeuk and Shin, Kijung},
  journal={arXiv preprint arXiv:2510.22215},
  year={2025}
}
