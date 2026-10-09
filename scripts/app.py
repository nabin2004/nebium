"""
Nebium Chess Transformer - Interactive Web UI
Supports all models across the Nebium family (Small, Base, Medium, Large, Local).
Deployable locally and on Hugging Face Spaces.
"""

import html
import json
import os
import sys
from pathlib import Path
import chess
import chess.svg
import gradio as gr
try:
    import spaces  # must come before torch
except ImportError:
    spaces = None


import torch

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if ROOT_DIR.name == "scripts":
    ROOT_DIR = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.transformer.nebium import Nebium
from src.data.tokenizer import ChessTokenizer

# Model Family Catalog
MODEL_CATALOG = {
    "Nebium-Small (117M) [Hugging Face]": {
        "repo_id": "nabin2004/nebium-small",
        "description": "117M parameter causal Transformer (12 layers, 768 dim, 12 heads)",
        "params": "117M",
        "default_weight": "model.pt",
        "default_config": "model_config.json",
        "default_tokenizer": "tokenizer.json",
    },
    "Nebium-Base (6M) [Hugging Face]": {
        "repo_id": "nabin2004/nebium",
        "description": "6M parameter base Transformer (6 layers, 512 dim, 8 heads)",
        "params": "6M",
        "default_weight": "model.pt",
        "default_config": "model_config.json",
        "default_tokenizer": "tokenizer.json",
    },
    "Nebium-Medium (345M) [Hugging Face]": {
        "repo_id": "nabin2004/nebium-medium",
        "description": "345M parameter causal Transformer (24 layers, 1024 dim, 16 heads)",
        "params": "345M",
        "default_weight": "model.pt",
        "default_config": "model_config.json",
        "default_tokenizer": "tokenizer.json",
    },
    "Nebium-Large (762M) [Hugging Face]": {
        "repo_id": "nabin2004/nebium-large",
        "description": "762M parameter flagship Transformer (36 layers, 1280 dim, 20 heads)",
        "params": "762M",
        "default_weight": "pytorch_model.bin",
        "fallback_config": {
            "vocab_size": 5000,
            "d_model": 1280,
            "n_heads": 20,
            "n_layers": 36,
            "max_seq_len": 1024,
            "dropout": 0.1,
            "positional_encoding": "rope",
            "activation": "swiglu",
            "norm": "rmsnorm",
            "attention_type": "standard",
        },
        "default_tokenizer": "tokenizer.json",
    },
    "Nebium-Local (Checkpoint: best_model.pt)": {
        "local_ckpt": "best_model.pt",
        "local_config": "export/model_config.json",
        "local_tokenizer": "export/tokenizer.json",
        "description": "Local workspace checkpoint (best_model.pt)",
        "params": "~6M",
    },
    "Custom / Manual Configuration": {
        "description": "Custom local checkpoint path or custom Hugging Face repo ID",
        "params": "Custom",
    },
}

# Global cached model state
_CURRENT_MODEL = None
_CURRENT_TOKENIZER = None
_CURRENT_MODEL_KEY = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_and_extract_state_dict(checkpoint):
    """Robustly extracts state_dict across varying PyTorch checkpoint formats."""
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    clean = {}
    for k, v in state_dict.items():
        clean_key = k.replace("_orig_mod.", "").replace("module.", "")
        clean[clean_key] = v
    return clean


def load_model_instance(model_choice: str, custom_repo_or_ckpt: str = ""):
    """Loads or retrieves cached model and tokenizer for the selected Nebium tier."""
    global _CURRENT_MODEL, _CURRENT_TOKENIZER, _CURRENT_MODEL_KEY

    cache_key = f"{model_choice}::{custom_repo_or_ckpt}"
    if _CURRENT_MODEL is not None and _CURRENT_MODEL_KEY == cache_key:
        return _CURRENT_MODEL, _CURRENT_TOKENIZER, f"Using cached {_CURRENT_MODEL_KEY}"

    print(f"Loading model: {model_choice} on {_DEVICE}...")
    tokenizer = ChessTokenizer()
    model = None

    try:
        if model_choice == "Nebium-Local (Checkpoint: best_model.pt)":
            ckpt_path = ROOT_DIR / "best_model.pt"
            config_path = ROOT_DIR / "export" / "model_config.json"
            tok_path = ROOT_DIR / "export" / "tokenizer.json"

            if not ckpt_path.exists():
                raise FileNotFoundError(f"Local checkpoint {ckpt_path} not found.")

            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg.pop("_target_", None)
            else:
                cfg = {
                    "vocab_size": 136,
                    "d_model": 512,
                    "n_heads": 8,
                    "n_layers": 2,
                    "dropout": 0.1,
                    "max_seq_len": 512,
                    "positional_encoding": "rope",
                    "activation": "swiglu",
                    "norm": "rmsnorm",
                }

            if not tok_path.exists():
                cand = ROOT_DIR / "data" / "tokenizer" / "lichess_2013" / "tokenizer.json"
                tok_path = cand if cand.exists() else tok_path

            tokenizer.load(str(tok_path))
            model = Nebium(**cfg)
            raw_ckpt = torch.load(str(ckpt_path), map_location=_DEVICE, weights_only=False)
            state_dict = clean_and_extract_state_dict(raw_ckpt)
            model.load_state_dict(state_dict, strict=False)

        elif model_choice in MODEL_CATALOG and "repo_id" in MODEL_CATALOG[model_choice]:
            meta = MODEL_CATALOG[model_choice]
            repo_id = meta["repo_id"]
            from huggingface_hub import hf_hub_download

            # 1. Tokenizer
            tok_file = hf_hub_download(repo_id=repo_id, filename=meta["default_tokenizer"])
            tokenizer.load(tok_file)

            # 2. Config
            cfg = None
            if "default_config" in meta:
                try:
                    cfg_file = hf_hub_download(repo_id=repo_id, filename=meta["default_config"])
                    with open(cfg_file, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    cfg.pop("_target_", None)
                except Exception:
                    cfg = None

            if cfg is None:
                cfg = meta.get("fallback_config", {
                    "vocab_size": tokenizer.vocab_size,
                    "d_model": 768,
                    "n_heads": 12,
                    "n_layers": 12,
                    "max_seq_len": 1024,
                    "dropout": 0.1,
                    "positional_encoding": "rope",
                    "activation": "swiglu",
                    "norm": "rmsnorm",
                })

            model = Nebium(**cfg)

            # 3. Weights
            weight_file = hf_hub_download(repo_id=repo_id, filename=meta["default_weight"])
            raw_ckpt = torch.load(weight_file, map_location=_DEVICE, weights_only=False)
            state_dict = clean_and_extract_state_dict(raw_ckpt)
            model.load_state_dict(state_dict, strict=False)

        else:
            # Custom input
            custom_input = custom_repo_or_ckpt.strip()
            if not custom_input:
                raise ValueError("Please provide a valid custom Hugging Face repo ID or local checkpoint path.")

            if os.path.exists(custom_input):
                # Local path
                raw_ckpt = torch.load(custom_input, map_location=_DEVICE, weights_only=False)
                state_dict = clean_and_extract_state_dict(raw_ckpt)
                tok_cand = ROOT_DIR / "export" / "tokenizer.json"
                tokenizer.load(str(tok_cand))
                with open(ROOT_DIR / "export" / "model_config.json", "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg.pop("_target_", None)
                model = Nebium(**cfg)
                model.load_state_dict(state_dict, strict=False)
            else:
                # Hugging Face Repo ID
                from huggingface_hub import hf_hub_download
                repo_id = custom_input
                tok_file = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
                tokenizer.load(tok_file)
                cfg_file = hf_hub_download(repo_id=repo_id, filename="model_config.json")
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg.pop("_target_", None)
                model = Nebium(**cfg)
                w_file = hf_hub_download(repo_id=repo_id, filename="model.pt")
                raw_ckpt = torch.load(w_file, map_location=_DEVICE, weights_only=False)
                state_dict = clean_and_extract_state_dict(raw_ckpt)
                model.load_state_dict(state_dict, strict=False)

        model.to(_DEVICE)
        model.eval()

        _CURRENT_MODEL = model
        _CURRENT_TOKENIZER = tokenizer
        _CURRENT_MODEL_KEY = cache_key

        param_count = sum(p.numel() for p in model.parameters())
        status_msg = f"✅ Loaded {model_choice} ({param_count:,} parameters) on {_DEVICE}"
        return model, tokenizer, status_msg

    except Exception as exc:
        raise gr.Error(f"Model Loading Failed: {str(exc)}")


def render_board_svg(moves_str: str) -> tuple[str, str, str, str]:
    """
    Parses moves, validates them on a chess.Board, and produces an SVG visualization
    along with game status and SAN history.
    """
    board = chess.Board()
    moves = [m.strip() for m in moves_str.strip().split() if m.strip()]
    san_list = []
    last_move = None
    illegal_move = None

    for m in moves:
        try:
            move = chess.Move.from_uci(m)
            if move in board.legal_moves:
                san_list.append(board.san(move))
                board.push(move)
                last_move = move
            else:
                illegal_move = m
                break
        except Exception:
            illegal_move = m
            break

    # Build responsive SVG board
    svg = chess.svg.board(board=board, lastmove=last_move, size=400)
    svg_html = f"<div style='display:flex; justify-content:center; align-items:center; padding:10px;'>{svg}</div>"

    # Status info
    turn = "White" if board.turn == chess.WHITE else "Black"
    status_parts = [f"**Turn:** {turn}", f"**Half-moves (Ply):** {len(moves)}"]
    if board.is_checkmate():
        status_parts.append("🔴 **Checkmate!**")
    elif board.is_stalemate():
        status_parts.append("🟡 **Stalemate (Draw)**")
    elif board.is_check():
        status_parts.append("⚠️ **Check!**")
    else:
        status_parts.append("🟢 **In Progress**")

    if illegal_move:
        status_parts.append(f"❌ *Stopped at illegal move:* `{illegal_move}`")

    status_markdown = " | ".join(status_parts)
    fen_display = board.fen()

    # Formatted SAN move list (e.g. 1. e4 e5 2. Nf3 Nc6)
    san_pairs = []
    for i in range(0, len(san_list), 2):
        move_num = (i // 2) + 1
        w_move = san_list[i]
        b_move = san_list[i + 1] if i + 1 < len(san_list) else ""
        san_pairs.append(f"{move_num}. {w_move} {b_move}".strip())
    san_display = "  ".join(san_pairs) if san_pairs else "(Starting position)"

    return svg_html, status_markdown, fen_display, san_display


def make_legal_tokens_filter(tokenizer: ChessTokenizer):
    """Creates a callable that restricts logits to strictly legal chess moves."""
    vocab = tokenizer.tokenizer.get_vocab()

    def legal_tokens_filter(token_ids):
        seq_str = tokenizer.decode(token_ids).strip()
        board = chess.Board()
        moves = seq_str.split()
        for m in moves:
            try:
                board.push_uci(m)
            except Exception:
                return None

        if board.is_game_over():
            return [tokenizer.eos_id]

        legal_ucis = {m.uci() for m in board.legal_moves}
        allowed = []
        for word, tid in vocab.items():
            w = word.strip()
            if not w:
                continue
            first_m = w.split()[0]
            if first_m in legal_ucis:
                allowed.append(tid)

        return allowed if allowed else None

    return legal_tokens_filter


def predict_moves(
    model_choice: str,
    custom_repo_or_ckpt: str,
    prompt_str: str,
    max_moves: int,
    temperature: float,
    top_k: int,
    top_p: float,
    enforce_legal: bool,
):
    """Runs autoregressive generation using the selected Nebium model."""
    model, tokenizer, status = load_model_instance(model_choice, custom_repo_or_ckpt)
    device = next(model.parameters()).device

    prompt_clean = prompt_str.strip()
    if prompt_clean:
        encoded = tokenizer.encode(prompt_clean)
        input_ids = [tokenizer.bos_id] + encoded
    else:
        input_ids = [tokenizer.bos_id]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(input_tensor, device=device)

    legal_fn = make_legal_tokens_filter(tokenizer) if enforce_legal else None

    with torch.no_grad():
        output = model.generate(
            input_tensor,
            mask,
            max_new_tokens=int(max_moves),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            legal_tokens_fn=legal_fn,
        )

    output_ids = output[0].tolist()
    if output_ids and output_ids[0] == tokenizer.bos_id:
        output_ids = output_ids[1:]

    generated_full = tokenizer.decode(output_ids).strip()
    # Filter special tokens
    for spec in ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]:
        generated_full = generated_full.replace(spec, "")
    generated_full = " ".join(generated_full.split())

    svg_html, status_md, fen, san = render_board_svg(generated_full)
    return generated_full, svg_html, status_md, fen, san, status


def step_single_move(
    model_choice: str,
    custom_repo_or_ckpt: str,
    current_moves: str,
    temperature: float,
    top_k: int,
    top_p: float,
    enforce_legal: bool,
):
    """Generates exactly one next move and appends it to the sequence."""
    return predict_moves(
        model_choice=model_choice,
        custom_repo_or_ckpt=custom_repo_or_ckpt,
        prompt_str=current_moves,
        max_moves=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        enforce_legal=enforce_legal,
    )


def undo_move(current_moves: str):
    """Pops the last move from the sequence."""
    moves = current_moves.strip().split()
    if moves:
        moves.pop()
    new_seq = " ".join(moves)
    svg_html, status_md, fen, san = render_board_svg(new_seq)
    return new_seq, svg_html, status_md, fen, san


# Build the Gradio Interface
custom_css = """
.board-container {
    display: flex;
    justify-content: center;
    align-items: center;
    background: #1e1e24;
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
.stat-box {
    background: #2a2a36;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
"""

with gr.Blocks(title="Nebium Chess Transformer") as demo:
    gr.Markdown(
        """
        # ♟️ Nebium Chess Transformer Explorer
        Interactive visual playground for the **Nebium** causal Transformer family.
        Inspect move sequences, play against the model, enforce chess move legality, and evaluate scaling tiers.
        """
    )

    with gr.Row():
        # Left Column: Model Selection & Parameters
        with gr.Column(scale=4):
            gr.Markdown("### 🏛️ Model Family Selection")
            model_selector = gr.Dropdown(
                choices=list(MODEL_CATALOG.keys()),
                value="Nebium-Small (117M) [Hugging Face]",
                label="Nebium Family Tier",
                info="Choose from published Hugging Face tiers or local checkpoints.",
            )

            custom_input = gr.Textbox(
                label="Custom Hugging Face Repo ID or Local .pt Checkpoint Path",
                placeholder="e.g. nabin2004/nebium-small or best_model.pt",
                visible=False,
            )

            model_status_badge = gr.Markdown("ℹ️ Model will load automatically on first generation.")

            def on_model_change(choice):
                show_custom = (choice == "Custom / Manual Configuration")
                info = MODEL_CATALOG.get(choice, {}).get("description", "")
                return gr.update(visible=show_custom), f"ℹ️ *{info}*"

            model_selector.change(on_model_change, inputs=[model_selector], outputs=[custom_input, model_status_badge])

            load_btn = gr.Button("⚡ Preload Model", variant="secondary")

            gr.Markdown("### 🎛️ Generation Controls")
            with gr.Row():
                temp_slider = gr.Slider(minimum=0.0, maximum=1.5, value=0.7, step=0.05, label="Temperature (Softmax)")
                top_k_slider = gr.Slider(minimum=0, maximum=50, value=20, step=1, label="Top-K Filter")
            with gr.Row():
                top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-P (Nucleus)")
                max_moves_slider = gr.Slider(minimum=1, maximum=40, value=6, step=1, label="Rollout Move Count")

            enforce_legal_chk = gr.Checkbox(
                value=True,
                label="🛡️ Enforce Legal Moves Only",
                info="Applies real-time move validation mask to eliminate illegal moves.",
            )

            gr.Markdown("### 📖 Standard Opening Presets")
            with gr.Row():
                btn_ruy = gr.Button("Ruy Lopez", size="sm")
                btn_sicilian = gr.Button("Sicilian", size="sm")
                btn_qg = gr.Button("Queen's Gambit", size="sm")
            with gr.Row():
                btn_french = gr.Button("French", size="sm")
                btn_italian = gr.Button("Italian", size="sm")
                btn_clear = gr.Button("🔄 Reset Board", size="sm", variant="stop")

        # Right Column: Visual Chessboard & Interactive Play
        with gr.Column(scale=6):
            gr.Markdown("### ♟️ Live Chessboard")
            init_svg, init_status, init_fen, init_san = render_board_svg("")
            board_display = gr.HTML(value=init_svg, elem_classes=["board-container"])
            status_display = gr.Markdown(value=init_status)

            with gr.Row():
                step_btn = gr.Button("▶️ Generate Next Move (1 Ply)", variant="primary", scale=2)
                gen_seq_btn = gr.Button("⏩ Generate Sequence", variant="secondary", scale=2)
                undo_btn = gr.Button("↩️ Undo Move", scale=1)

            moves_input = gr.Textbox(
                label="UCI Move Sequence",
                value="",
                placeholder="e.g. e2e4 e7e5 g1f3 b8c6",
                lines=2,
            )

            with gr.Accordion("SAN Notation & FEN Details", open=True):
                san_display = gr.Textbox(label="Standard Algebraic Notation (SAN)", value=init_san, interactive=False)
                fen_display = gr.Textbox(label="Board FEN", value=init_fen, interactive=False)

    # Event handlers
    load_btn.click(
        fn=lambda m, c: load_model_instance(m, c)[2],
        inputs=[model_selector, custom_input],
        outputs=[model_status_badge],
    )

    # Update board when typing moves manually
    moves_input.change(
        fn=render_board_svg,
        inputs=[moves_input],
        outputs=[board_display, status_display, fen_display, san_display],
    )

    # Step single move
    step_btn.click(
        fn=step_single_move,
        inputs=[
            model_selector,
            custom_input,
            moves_input,
            temp_slider,
            top_k_slider,
            top_p_slider,
            enforce_legal_chk,
        ],
        outputs=[moves_input, board_display, status_display, fen_display, san_display, model_status_badge],
    )

    # Generate sequence
    gen_seq_btn.click(
        fn=predict_moves,
        inputs=[
            model_selector,
            custom_input,
            moves_input,
            max_moves_slider,
            temp_slider,
            top_k_slider,
            top_p_slider,
            enforce_legal_chk,
        ],
        outputs=[moves_input, board_display, status_display, fen_display, san_display, model_status_badge],
    )

    # Undo
    undo_btn.click(
        fn=undo_move,
        inputs=[moves_input],
        outputs=[moves_input, board_display, status_display, fen_display, san_display],
    )

    # Presets
    btn_ruy.click(fn=lambda: "e2e4 e7e5 g1f3 b8c6 f1b5", outputs=[moves_input])
    btn_sicilian.click(fn=lambda: "e2e4 c7c5", outputs=[moves_input])
    btn_qg.click(fn=lambda: "d2d4 d7d5 c2c4", outputs=[moves_input])
    btn_french.click(fn=lambda: "e2e4 e7e6", outputs=[moves_input])
    btn_italian.click(fn=lambda: "e2e4 e7e5 g1f3 b8c6 f1c4", outputs=[moves_input])
    btn_clear.click(fn=lambda: "", outputs=[moves_input])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch Nebium Gradio UI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Generate public gradio.live share link")
    args = parser.parse_args()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
