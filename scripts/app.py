import gradio as gr
import torch
import hydra
from hydra.utils import instantiate
from hydra import compose, initialize
from src.data.prepare import get_tokenizer

# Global state to cache the loaded model
_current_model = None
_current_config_name = None
_current_ckpt_path = None
_tokenizer = None

def load_model(config_name, ckpt_path):
    global _current_model, _current_config_name, _current_ckpt_path, _tokenizer
    
    if _current_model is not None and _current_config_name == config_name and _current_ckpt_path == ckpt_path:
        return _current_model, _tokenizer
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            initialize(version_base=None, config_path="../configs")
            
        cfg = compose(config_name="config", overrides=[f"model={config_name}"])
            
        _tokenizer = get_tokenizer(cfg)
        model = instantiate(cfg.model)
        
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Successfully loaded weights from {ckpt_path}")
        except FileNotFoundError:
            print(f"Warning: Checkpoint {ckpt_path} not found. Using randomly initialized weights.")
        except Exception as e:
            print(f"Warning: Failed to load weights from {ckpt_path}: {e}. Using randomly initialized weights.")
            
        model.to(device)
        model.eval()
        
        _current_model = model
        _current_config_name = config_name
        _current_ckpt_path = ckpt_path
        
        return model, _tokenizer
    except Exception as e:
        raise gr.Error(f"Failed to load model: {str(e)}")

def generate_moves(config_name, ckpt_path, prompt, max_moves, temperature, top_k, top_p):
    model, tokenizer = load_model(config_name, ckpt_path)
    device = next(model.parameters()).device
    
    prompt = prompt.strip()
    if prompt:
        input_ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
    else:
        input_ids = [tokenizer.bos_id]
        
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor, device=device)
    
    with torch.no_grad():
        output = model.generate(
            input_tensor,
            attention_mask,
            max_new_tokens=int(max_moves),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p)
        )
    
    output_ids = output[0].tolist()
    # Strip BOS if present at start
    if output_ids[0] == tokenizer.bos_id:
        output_ids = output_ids[1:]
        
    decoded = tokenizer.decode(output_ids)
    return decoded

with gr.Blocks(title="Nebium Chess Transformer UI") as demo:
    gr.Markdown("# ♟️ Nebium Chess Transformer")
    gr.Markdown("Interactive UI for testing the Nebium autoregressive transformer family.")
    
    with gr.Row():
        with gr.Column(scale=1):
            config_dropdown = gr.Dropdown(
                choices=["nebium_stub", "nebium_base"],
                value="nebium_stub",
                label="Model Configuration"
            )
            ckpt_input = gr.Textbox(
                value="best_model.pt",
                label="Checkpoint Path"
            )
            max_moves_slider = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Max Moves")
            temp_slider = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="Temperature")
            top_k_slider = gr.Slider(minimum=0, maximum=100, value=0, step=1, label="Top K")
            top_p_slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="Top P")
            
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                placeholder="e.g. e2e4 e7e5 g1f3",
                label="Initial Moves",
                lines=2
            )
            generate_btn = gr.Button("Generate Moves", variant="primary")
            output_text = gr.Textbox(
                label="Generated Sequence",
                lines=10,
                interactive=False
            )
            
    generate_btn.click(
        fn=generate_moves,
        inputs=[config_dropdown, ckpt_input, prompt_input, max_moves_slider, temp_slider, top_k_slider, top_p_slider],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
