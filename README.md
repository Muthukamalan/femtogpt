# FemtoGPT
A tiny, super minimal Chat Model.

<!-- https://github.com/huggingface/transformers/blob/resnet_with_variants/examples/research_projects/codeparrot/scripts/codeparrot_training.py -->
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


```log
tensorboard --logdir=trainer_output/runs/
```
![Logs](./assets/logs.png)
