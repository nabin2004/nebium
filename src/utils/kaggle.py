import os
import sys
from pathlib import Path


def is_kaggle_environment() -> bool:
    """Detects whether code is executing inside a Kaggle notebook/script environment."""
    return (
        os.path.exists("/kaggle")
        or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "KAGGLE_URL_BASE" in os.environ
    )


def setup_kaggle_env(working_dir: str | Path = "/kaggle/working") -> dict[str, bool]:
    """
    Initializes and configures the environment when running on Kaggle.
    - Loads WANDB_API_KEY and HF_TOKEN from Kaggle Secrets if not already present.
    - Prepares working and output directories.
    - Returns a status dictionary of available services.
    """
    status = {
        "is_kaggle": is_kaggle_environment(),
        "wandb_configured": bool(os.environ.get("WANDB_API_KEY")),
        "hf_configured": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    }

    if not status["is_kaggle"]:
        return status

    print("\n" + "=" * 60)
    print(" [Kaggle Setup] Detected Kaggle execution environment.")
    print("=" * 60)

    # Attempt to load secrets using Kaggle UserSecretsClient
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        user_secrets = UserSecretsClient()

        # WandB API Key
        if not status["wandb_configured"]:
            for secret_name in ("WANDB_API_KEY", "wandb_api_key", "wandb_key", "WANDB_KEY"):
                try:
                    val = user_secrets.get_secret(secret_name)
                    if val:
                        os.environ["WANDB_API_KEY"] = val.strip()
                        status["wandb_configured"] = True
                        break
                except Exception:
                    pass

        # Hugging Face Token
        if not status["hf_configured"]:
            for secret_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "huggingface_token", "HF_API_KEY"):
                try:
                    val = user_secrets.get_secret(secret_name)
                    if val:
                        token_str = val.strip()
                        os.environ["HF_TOKEN"] = token_str
                        os.environ["HUGGING_FACE_HUB_TOKEN"] = token_str
                        status["hf_configured"] = True
                        break
                except Exception:
                    pass

    except (ImportError, Exception) as exc:
        print(f" [Kaggle Setup] Note: UserSecretsClient unavailable or error: {exc}")

    # Ensure working output directories exist
    target_dir = Path(working_dir)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "outputs").mkdir(parents=True, exist_ok=True)
        (target_dir / "data").mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f" [Kaggle Setup] Warning creating working dirs: {exc}")

    print(f"  * WandB API Key : {'✓ Connected' if status['wandb_configured'] else '✗ Not found (set WANDB_API_KEY secret)'}")
    print(f"  * Hugging Face  : {'✓ Connected' if status['hf_configured'] else '✗ Not found (set HF_TOKEN secret)'}")
    print(f"  * Working Dir   : {target_dir}")
    print("=" * 60 + "\n")

    return status
