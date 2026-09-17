import argparse
from huggingface_hub import HfApi
import os

def main():
    parser = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, default="nabin2004/nebium", help="HF Repo ID")
    parser.add_argument("--token", type=str, default=None, help="HF Token (optional if logged in)")
    args = parser.parse_args()

    api = HfApi()
    
    # Create repo if not exists
    try:
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, token=args.token)
        print(f"Repository {args.repo_id} is ready.")
    except Exception as e:
        print(f"Warning/Error creating repo: {e}")

    # Files to upload
    files = {
        "best_model.pt": "best_model.pt",
        "data/tokenizer/fixture/tokenizer.json": "tokenizer.json",
        "MODEL_CARD.md": "README.md",
    }
    
    for local_path, repo_path in files.items():
        if os.path.exists(local_path):
            print(f"Uploading {local_path} to {repo_path}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=args.repo_id,
                repo_type="model",
                token=args.token
            )
        else:
            print(f"File not found, skipping: {local_path}")
            
    print("Push complete! Check your model on Hugging Face Hub.")

if __name__ == "__main__":
    main()
