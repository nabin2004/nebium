"""
Nebium Chess Transformer - Root Entrypoint for Gradio Web UI
Delegates to scripts/app.py
"""

from scripts.app import demo

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
