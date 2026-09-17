# Automated Evaluation Pipeline

Nebium utilizes an extensive, automated evaluation pipeline to rigorously assess the model's chess capabilities beyond next-token loss. 

## The Master Script
`scripts/generate_paper_report.py` acts as the orchestrator. It runs all individual evaluation scripts, collects their JSON outputs, generates matplotlib charts (`paper_assets/`), and compiles a highly structured `FINAL_PAPER_REPORT.md` markdown file.

This script is automatically executed at the end of the `train.py` loop if `cfg.training.generate_report` is enabled.

## Individual Evaluation Scripts

### 1. Self-Play Elo Estimation (`scripts/eval_self_play.py`)
- Pits Nebium against the Stockfish engine (at a fixed depth, e.g., 10).
- Plays a set number of games.
- Uses a Bayesian Elo calculation based on win/loss/draw rates against the baseline engine to estimate the model's true rating.

### 2. Opening Book Compliance (`scripts/eval_openings.py`)
- Evaluates if the model correctly plays established theoretical opening lines (e.g., Ruy Lopez, Sicilian Najdorf).
- Feeds the model the first $N$ moves of an opening and verifies if the predicted next move adheres to theory.

### 3. Endgame Conversion (`scripts/eval_endgames.py`)
- Tests whether the model can convert highly winning positions (e.g., King + Queen vs. King) against Stockfish.
- Important for evaluating if the model actually understands goal-oriented play (checkmating) rather than just shuffling pieces.

### 4. Blunder and Loop Analysis (`scripts/analyze_errors.py`)
- Evaluates the model's games using Stockfish evaluation (Centipawns).
- **Blunders**: Measures how often the model makes a move that drops the evaluation by >2.0 pawns.
- **Loops**: Tracks how often the model gets stuck in 3-fold repetitions or 50-move rules (a common failure mode of ungrounded causal models).

### 5. Memorization & Data Leakage (`scripts/analyze_memorization.py`)
- Computes overlapping $N$-grams between the training dataset and the testing/puzzle datasets.
- Assesses if the model is actually generalizing or simply regurgitating deeply memorized game sequences.

### 6. Inference Latency (`scripts/benchmark_latency.py`)
- Compares the time-to-first-token for various decoding strategies: Greedy, Top-K, Beam Search, and Legal-Filtered decoding.
