#!/usr/bin/env python3
"""
Script to push trained model checkpoints to Hugging Face Hub.

Usage:
    python push_to_huggingface.py --repo-id YOUR_USERNAME/YOUR_REPO_NAME --token YOUR_HF_TOKEN

Options:
    --checkpoint: Specific checkpoint to upload (e.g., "checkpoint-40000"). 
                  If not specified, uploads the latest checkpoint.
    --all: Upload all checkpoints
    --private: Make the repository private (default: True)
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from tqdm import tqdm


def get_checkpoints(base_dir: str) -> list[Path]:
    """Get all checkpoint directories sorted by step number."""
    checkpoints_path = Path(base_dir)
    checkpoints = []
    
    for item in checkpoints_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step = int(item.name.split("-")[1])
                checkpoints.append((step, item))
            except (ValueError, IndexError):
                continue
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    return [cp[1] for cp in checkpoints]


def get_latest_checkpoint(base_dir: str) -> Path | None:
    """Get the latest checkpoint directory."""
    checkpoints = get_checkpoints(base_dir)
    return checkpoints[-1] if checkpoints else None


def push_checkpoint_to_hub(
    checkpoint_path: Path,
    repo_id: str,
    token: str,
    private: bool = True,
    subfolder: str | None = None,
    commit_message: str | None = None,
) -> str:
    """Push a checkpoint folder to Hugging Face Hub."""
    
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="model",
        )
        print(f"✓ Repository '{repo_id}' is ready (private={private})")
    except Exception as e:
        print(f"Note: {e}")
    
    # Prepare commit message
    if commit_message is None:
        commit_message = f"Upload {checkpoint_path.name}"
    
    # Upload the folder
    print(f"📤 Uploading {checkpoint_path.name} to {repo_id}...")
    print(f"   Path: {checkpoint_path}")
    
    # Files to ignore (optional - uncomment to skip optimizer states to save space)
    ignore_patterns = [
        # "optimizer.pt",  # Uncomment to skip optimizer (saves ~2GB per checkpoint)
        # "rng_state.pth",  # Uncomment to skip RNG state
        "__pycache__",
        "*.pyc",
    ]
    
    url = upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=repo_id,
        token=token,
        path_in_repo=subfolder if subfolder else "",  # Root or subfolder
        commit_message=commit_message,
        ignore_patterns=ignore_patterns,
    )
    
    print(f"✓ Successfully uploaded to: {url}")
    return url


def main():
    parser = argparse.ArgumentParser(
        description="Push trained model checkpoints to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Hugging Face API token (get from https://huggingface.co/settings/tokens)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint to upload (e.g., 'checkpoint-40000'). "
             "If not specified, uploads the latest checkpoint.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Upload all checkpoints (each in its own subfolder)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Make the repository private (default: True)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the repository public",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints_all/neutts-vietnamese",
        help="Path to checkpoints directory (default: checkpoints_all/neutts-vietnamese)",
    )
    parser.add_argument(
        "--skip-optimizer",
        action="store_true",
        help="Skip uploading optimizer.pt files to save space (~2GB per checkpoint)",
    )
    
    args = parser.parse_args()
    
    # Determine if private
    is_private = not args.public
    
    # Get the base directory
    script_dir = Path(__file__).parent.resolve()
    checkpoints_dir = script_dir / args.checkpoints_dir
    
    if not checkpoints_dir.exists():
        print(f"❌ Error: Checkpoints directory not found: {checkpoints_dir}")
        return 1
    
    print(f"🔍 Looking for checkpoints in: {checkpoints_dir}")
    
    if args.all:
        # Upload all checkpoints
        checkpoints = get_checkpoints(str(checkpoints_dir))
        if not checkpoints:
            print("❌ No checkpoints found!")
            return 1
        
        print(f"📦 Found {len(checkpoints)} checkpoints to upload")
        
        for checkpoint in tqdm(checkpoints, desc="Uploading checkpoints"):
            push_checkpoint_to_hub(
                checkpoint_path=checkpoint,
                repo_id=args.repo_id,
                token=args.token,
                private=is_private,
                subfolder=checkpoint.name,  # Each checkpoint in its own folder
                commit_message=f"Add {checkpoint.name}",
            )
        
        print(f"\n✅ All {len(checkpoints)} checkpoints uploaded successfully!")
        
    elif args.checkpoint:
        # Upload specific checkpoint
        checkpoint_path = checkpoints_dir / args.checkpoint
        if not checkpoint_path.exists():
            print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
            print(f"   Available checkpoints: {[cp.name for cp in get_checkpoints(str(checkpoints_dir))]}")
            return 1
        
        push_checkpoint_to_hub(
            checkpoint_path=checkpoint_path,
            repo_id=args.repo_id,
            token=args.token,
            private=is_private,
        )
        
    else:
        # Upload latest checkpoint
        checkpoint_path = get_latest_checkpoint(str(checkpoints_dir))
        if checkpoint_path is None:
            print("❌ No checkpoints found!")
            return 1
        
        print(f"📦 Using latest checkpoint: {checkpoint_path.name}")
        
        push_checkpoint_to_hub(
            checkpoint_path=checkpoint_path,
            repo_id=args.repo_id,
            token=args.token,
            private=is_private,
        )
    
    print(f"\n🎉 Done! View your model at: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    exit(main())
