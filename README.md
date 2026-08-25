

## Design Principles:
1. Configuration over code
2. Pluggable components
3. Dataset-agnostic interface
4. Training loop decoupling
5. Extensibility for future models

## Implementation Order:
1. Tokenizer
2. PyTorch Dataset + DataLoader
3. Embedding layer
4. Positional Encoding
5. MHSA
6. FFN
7. Transformer Block (Combine MHSA + FFN)
8. Avengers Assemble here (Assemble all components into Nebium model)
9. Prepare the training loop