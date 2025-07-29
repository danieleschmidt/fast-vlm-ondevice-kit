#!/usr/bin/env python3
"""
Download FastVLM model checkpoints for conversion.

This script downloads pre-trained FastVLM models from various sources
and prepares them for Core ML conversion.
"""

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    "fast-vlm-tiny": {
        "url": "https://example.com/models/fast-vlm-tiny.pth",
        "sha256": "placeholder_hash_tiny",
        "size_mb": 98,
        "description": "Smallest model for real-time applications"
    },
    "fast-vlm-base": {
        "url": "https://example.com/models/fast-vlm-base.pth", 
        "sha256": "placeholder_hash_base",
        "size_mb": 412,
        "description": "Balanced model for general use"
    },
    "fast-vlm-large": {
        "url": "https://example.com/models/fast-vlm-large.pth",
        "sha256": "placeholder_hash_large", 
        "size_mb": 892,
        "description": "Largest model for maximum accuracy"
    }
}


def download_file(url: str, filepath: Path, expected_size: int = None) -> None:
    """Download file with progress bar and validation."""
    logger.info(f"Downloading {url} to {filepath}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
                    
    if expected_size and filepath.stat().st_size != expected_size * 1024 * 1024:
        logger.warning(f"File size mismatch for {filepath}")


def verify_checksum(filepath: Path, expected_hash: str) -> bool:
    """Verify file SHA256 checksum."""
    logger.info(f"Verifying checksum for {filepath}")
    
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    actual_hash = sha256_hash.hexdigest()
    
    if actual_hash != expected_hash:
        logger.error(f"Checksum mismatch: expected {expected_hash}, got {actual_hash}")
        return False
    
    logger.info("Checksum verification passed")
    return True


def download_model(model_name: str, output_dir: Path = None) -> None:
    """Download and verify a specific model."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_info = MODELS[model_name]
    
    if output_dir is None:
        output_dir = Path("checkpoints")
    
    filepath = output_dir / f"{model_name}.pth"
    
    # Skip if already exists and verified
    if filepath.exists():
        logger.info(f"Model {model_name} already exists at {filepath}")
        if verify_checksum(filepath, model_info["sha256"]):
            logger.info("Using existing verified model")
            return
        else:
            logger.warning("Existing model failed verification, re-downloading")
            filepath.unlink()
    
    # Download the model
    download_file(
        model_info["url"], 
        filepath, 
        model_info["size_mb"]
    )
    
    # Verify checksum
    if not verify_checksum(filepath, model_info["sha256"]):
        filepath.unlink()
        raise RuntimeError(f"Downloaded model {model_name} failed verification")
    
    logger.info(f"Successfully downloaded {model_name} to {filepath}")


def list_models() -> None:
    """List available models."""
    print("Available FastVLM models:")
    print()
    
    for name, info in MODELS.items():
        print(f"  {name}")
        print(f"    Size: {info['size_mb']}MB")
        print(f"    Description: {info['description']}")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download FastVLM model checkpoints"
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["all"],
        required=True,
        help="Model to download"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="checkpoints",
        help="Output directory for models"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    try:
        if args.model == "all":
            for model_name in MODELS.keys():
                download_model(model_name, args.output_dir)
        else:
            download_model(args.model, args.output_dir)
            
        logger.info("Download completed successfully")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()