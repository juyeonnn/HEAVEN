# HEAVEN: Hybrid-Vector Retrieval for Visually Rich Documents

Official Repository for our paper "Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy"


## 0. ViMDoc Benchmark 
**ViMDoc** (Visually-rich Long Multi-Document Retrieval Benchmark) for evaluating visual document
retrieval under both **multi-document** and **long-document** settings. ViMDoc is available at `benchmark/ViMDoc`.

Other benchmarks are available both in our repository (`benchmark/OpenDocVQA` |`benchmark/ViDoSeek` | `benchmark/M3DocVQA`) and from original sources ( [OpenDocVQA](https://huggingface.co/datasets/NTT-hil-insight/OpenDocVQA) | [ViDoSeek](https://huggingface.co/datasets/autumncc/ViDoSeek) | [M3DocVQA](https://github.com/bloomberg/m3docrag) )

###  Format
All benchmarks contain `test.json` with queries and ground truth document IDs:
```json
{
    "id": "<query_id>",
    "query": "<query_text>",
    "doc_ids": [
        "<document_id>"
    ]
}
```
Sampled document pages are stored in `sampled_pages/`

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
