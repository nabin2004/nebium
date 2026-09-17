import argparse
import json
from pathlib import Path

def get_ngrams(sequence: str, n: int) -> set:
    moves = sequence.strip().split()
    if len(moves) < n:
        return set()
    return set(tuple(moves[i:i+n]) for i in range(len(moves) - n + 1))

def main():
    parser = argparse.ArgumentParser(description="Analyze n-gram overlap between training data and test sets.")
    parser.add_argument("--train_moves", type=str, default="data/processed/fixture/moves.txt", help="Path to training moves.txt")
    parser.add_argument("--test_puzzles", type=str, default="data/fixtures/puzzles.jsonl", help="Path to test puzzles JSONL")
    parser.add_argument("--n", type=int, default=10, help="N-gram length to compute overlap")
    args = parser.parse_args()

    train_path = Path(args.train_moves)
    if not train_path.exists():
        print(f"Train file not found: {train_path}")
        return

    test_path = Path(args.test_puzzles)
    if not test_path.exists():
        print(f"Test file not found: {test_path}")
        return

    print(f"Computing {args.n}-grams for training data...")
    train_ngrams = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            train_ngrams.update(get_ngrams(line, args.n))

    print(f"Loaded {len(train_ngrams)} unique {args.n}-grams from training data.")

    print(f"Computing {args.n}-grams for test data...")
    test_ngrams = set()
    test_sequences = 0
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            puzzle = json.loads(line)
            prompt = puzzle.get("prompt", "")
            full_solution = puzzle.get("full_solution", [])
            
            sequence = prompt
            if full_solution:
                sequence += " " + " ".join(full_solution)
                
            if sequence.strip():
                test_sequences += 1
                test_ngrams.update(get_ngrams(sequence, args.n))

    print(f"Loaded {len(test_ngrams)} unique {args.n}-grams from {test_sequences} test sequences.")

    overlap = train_ngrams.intersection(test_ngrams)
    print(f"Found {len(overlap)} overlapping {args.n}-grams.")
    
    if test_ngrams:
        leakage = len(overlap) / len(test_ngrams) * 100
        print(f"Data Leakage / Memorization Metric: {leakage:.2f}% of test {args.n}-grams are in the training set.")
    else:
        print("No test n-grams to compare.")

if __name__ == "__main__":
    main()
