# FemtoGPT
A tiny, super-minimal chat model built from scratch — tokenizer → transformer → training (base-model) → Inference API .

> Goal: Understand and implement every core component of an LLM pipeline in the simplest possible way.

## Features
- Custom **BPE tokenizer** (no external tokenizer dependency)
- Minimal **Transformer architecture** (RMSNorm, RoPE, SwiGLU)
- Lightweight **80M parameter model**
- Hugging Face **Trainer-based training pipeline**
- Simple **web chat UI**
- Dockerized setup for easy deployment


## Roadmap

1. Tokenization
    - [x] Add a simple BPE training implementation in Python.
    - [x] Add Regex Pretokenization.
    - [x] Upgrade BPE training to consider frequency deltas instead of recounting strategy.
    - [x] Add a simple BPE encoding/decoding function.
    - [x] Add support for special tokens.
    - [x] Train Tokenizer on TinyStories.
2. LLM Architecture
    - [x] Linear Layer
    - [x] Embedding Layer
    - [x] RMSNorm
    - [x] SwiGLU
    - [x] RoPE
    - [X] Attention
3. Train 80M Model
    - [X] PreTrain
    - [ ] MidTrain
    - [ ] PostTrain
4. Chat Interface
    - [ ] SFT
    - [ ] Alignment
    - [ ] Eval
    - [ ] LORA
    - [ ] QLORA
5. Inference
    - [ ] FastAPI

## Installation
To install the dependencies simply run the following command:
```sh
pip install -r requirements.txt
```
Before you run any of the scripts make sure you are logged in and can push to the hub:
```sh
huggingface-cli login
```


## Dataset
The source of the dataset is from the 🤗 Datasets Huggingface and it's streaming and get only text
- HuggingFaceTB/smollm-corpus
- databricks/databricks-dolly-15k
- Abirate/english_quotes
- b-mc2/sql-create-context
- squad
- tatsu-lab/alpaca

## Tokenizers
- HuggingFaceTB/SmolLM-135M
- Fully custom [BPE](https://github.com/Muthukamalan/TamilTokenizers) implementation
**Supports:**
    - Special tokens
    - Efficient encoding/decoding
    - Regex pre-tokenization

|Model        | Details                  |
|-------------|--------------------------|
|Architecture |	Decoder-only Transformer |
|Parameters   |	~80M                     |
|Positional   | Encoding	RoPE         |
|Normalization|	RMSNorm                  |
|Activation   |	SwiGLU                   |
|Attention    |	Multi-head self-attention|
|Attn Impl    | MHLA                     |
|Optimizer    | AdamW                    |
|Scheduler    | Cosine Scheduler with warmup|
|Logs         | Tensorboard              |
|Generation   | Beam search              |
| fp16        | True                     |


## trainer API
```sh
# using trainer API from huggingface
python train.py
```



```log
tensorboard --logdir=trainer_output/runs/
```
![Logs](./assets/logs.png)


```sh
# add .env file under app/backend/
# HF_TOKEN=
docker compose up 
```

![UI](./assets/webui.png)
