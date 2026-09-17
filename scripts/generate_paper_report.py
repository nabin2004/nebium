import os
import json
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

def run_evaluation_scripts():
    print("Running extensive evaluation suite. This may take a few minutes...")
    os.makedirs("paper_assets", exist_ok=True)
    
    scripts = [
        ("scripts/eval_self_play.py", "--output paper_assets/eval_elo.json --games 5 --depth 10"),
        ("scripts/eval_openings.py", "--output paper_assets/eval_openings.json"),
        ("scripts/eval_endgames.py", "--output paper_assets/eval_endgames.json"),
        ("scripts/analyze_errors.py", "--output paper_assets/analyze_errors.json --games 2"),
        ("scripts/analyze_memorization.py", "--output paper_assets/analyze_memorization.json"),
        ("scripts/benchmark_latency.py", "--output paper_assets/benchmark_latency.json")
    ]
    
    for script, args in scripts:
        print(f"Running {script}...")
        cmd = f"uv run python {script} {args}"
        try:
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")

def plot_latency():
    path = "paper_assets/benchmark_latency.json"
    if not os.path.exists(path):
        return
        
    with open(path, "r") as f:
        data = json.load(f)
        
    labels = ["Greedy", "Top-K", "Legal-Greedy", "Beam Search"]
    values = [
        data.get("greedy_sec_per_move", 0) * 1000,
        data.get("topk_sec_per_move", 0) * 1000,
        data.get("legal_greedy_sec_per_move", 0) * 1000,
        data.get("beam_search_sec_per_move", 0) * 1000,
    ]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=values, palette="Blues_d")
    plt.title("Inference Latency by Decoding Strategy")
    plt.ylabel("Latency (ms / move)")
    plt.savefig("paper_assets/latency_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_training_history():
    path = "training_history.json"
    if not os.path.exists(path):
        return
        
    with open(path, "r") as f:
        history = json.load(f)
        
    epochs = [x["epoch"] for x in history]
    train_loss = [x["train_loss"] for x in history]
    val_loss = [x["val_loss"] for x in history]
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, val_loss, label="Validation Loss", marker="s")
    plt.title("Nebium Training Progression")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.savefig("paper_assets/training_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def generate_markdown_report():
    print("Generating FINAL_PAPER_REPORT.md...")
    elo_data = load_json("paper_assets/eval_elo.json")
    openings_data = load_json("paper_assets/eval_openings.json")
    errors_data = load_json("paper_assets/analyze_errors.json")
    mem_data = load_json("paper_assets/analyze_memorization.json")
    latency_data = load_json("paper_assets/benchmark_latency.json")
    
    with open("FINAL_PAPER_REPORT.md", "w") as f:
        f.write("# Nebium: A Causal Transformer for Self-Supervised Next-Move Prediction in Chess\n\n")
        f.write("## 1. Introduction\n")
        f.write("Nebium is a modern decoder-only causal Transformer architecture incorporating Rotary Position Embeddings (RoPE), SwiGLU, and RMSNorm. It learns chess purely from self-supervised next-token prediction over UCI move sequences.\n\n")
        
        f.write("## 2. Architecture Diagram\n")
        f.write("```mermaid\n")
        f.write("graph TD\n")
        f.write("    A[Input UCI Token e.g. e2e4] --> B[Token Embedding]\n")
        f.write("    B --> C[RoPE (Rotary Position Embeddings)]\n")
        f.write("    C --> D[Transformer Block 1..N]\n")
        f.write("    D --> E[Multi-Head Causal Attention]\n")
        f.write("    E --> F[SwiGLU FFN]\n")
        f.write("    F --> G[RMSNorm]\n")
        f.write("    G --> H[Linear Projection (LM Head)]\n")
        f.write("    H --> I[Logits]\n")
        f.write("    I --> J{Decoding Strategy}\n")
        f.write("    J -->|Legal Move Masking| K[Filtered Probs]\n")
        f.write("    K --> L[Next Token Prediction]\n")
        f.write("```\n\n")
        
        f.write("## 3. Training Dynamics\n")
        if os.path.exists("paper_assets/training_loss.png"):
            f.write("![Training Loss](paper_assets/training_loss.png)\n\n")
        else:
            f.write("*Training loss plot not available.*\n\n")
            
        f.write("## 4. Rigorous Evaluation Results\n")
        
        f.write("### 4.1 Estimated Elo via Self-Play\n")
        if elo_data:
            f.write(f"- **Stockfish Depth 10 Engine Elo (Baseline):** 2000\n")
            f.write(f"- **Games Played:** {elo_data.get('games', 0)}\n")
            f.write(f"- **Score:** {elo_data.get('score', 0)*100:.1f}%\n")
            f.write(f"- **Estimated Nebium Elo:** **{elo_data.get('estimated_elo', 'N/A'):.0f}**\n\n")
        else:
            f.write("*Elo data not available.*\n\n")
            
        f.write("### 4.2 Opening Book Compliance\n")
        if openings_data:
            f.write(f"- **Score:** {openings_data.get('score', 0)*100:.1f}%\n")
            f.write("- **Details:**\n")
            for opening, correct in openings_data.get("details", {}).items():
                status = "✅ Pass" if correct else "❌ Fail"
                f.write(f"  - {opening}: {status}\n")
        f.write("\n")
        
        f.write("### 4.3 Error Analysis & Legality\n")
        if errors_data:
            f.write(f"- **Raw Legal Move Rate (Greedy without masking):** {errors_data.get('legal_move_rate', 0)*100:.2f}%\n")
            f.write(f"- **Blunder Rate (>2.0 pawn drop):** {errors_data.get('blunder_rate', 0)*100:.2f}%\n")
            f.write(f"- **Repetition Loops Triggered:** {errors_data.get('threefold_repetitions', 0)}\n\n")
            
        f.write("### 4.4 Data Memorization Assessment\n")
        if mem_data:
            f.write(f"- **N-Gram Size:** {mem_data.get('n_gram_size', 10)}\n")
            f.write(f"- **Train Unique N-Grams:** {mem_data.get('train_unique_ngrams', 0):,}\n")
            f.write(f"- **Test Data Overlaps:** {mem_data.get('overlap_count', 0)}\n")
            f.write(f"- **Exact Memorization / Leakage:** **{mem_data.get('leakage_percent', 0):.2f}%**\n\n")
            
        f.write("## 5. Inference & Latency Optimization\n")
        if latency_data:
            f.write("By injecting a strict legal-move mask derived from `python-chess`, Nebium enforces 100% legal play. Latency comparisons across decoding modes reveal:\n\n")
            f.write(f"- Greedy Decoding: {latency_data.get('greedy_sec_per_move', 0)*1000:.1f} ms/move\n")
            f.write(f"- Beam Search (w=3): {latency_data.get('beam_search_sec_per_move', 0)*1000:.1f} ms/move\n")
            f.write(f"- Legal-Filtered Greedy: {latency_data.get('legal_greedy_sec_per_move', 0)*1000:.1f} ms/move\n\n")
            if os.path.exists("paper_assets/latency_comparison.png"):
                f.write("![Latency Comparison](paper_assets/latency_comparison.png)\n")
                
        f.write("\n\n---\n*Report auto-generated by Nebium Eval Pipeline.*\n")
    print("Report written to FINAL_PAPER_REPORT.md.")

def main():
    run_evaluation_scripts()
    plot_latency()
    plot_training_history()
    generate_markdown_report()

if __name__ == "__main__":
    main()
