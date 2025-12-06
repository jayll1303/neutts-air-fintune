"""
Script to prepare Vietnamese dataset for NeuTTS-Air finetuning.
This script:
1. Reads metadata.csv
2. Encodes all audio files using NeuCodec (with batch processing)
3. Saves the encoded dataset as a pickle file

Optimized for large datasets (600k+ samples):
- No file existence check during metadata reading (handled during encoding)
- No sorting by file size (avoids 600k+ getsize calls)
- Parallel audio loading with ThreadPoolExecutor
"""

import os
import csv
import torch
import pickle
import numpy as np
from tqdm import tqdm
from librosa import load
from neucodec import NeuCodec
from concurrent.futures import ThreadPoolExecutor, as_completed

# NeuCodec downsampling ratio (16kHz / 50Hz = 320)
DOWNSAMPLE_RATIO = 320


def load_audio(path):
    """Load a single audio file. Returns (wav, length) or (None, 0) on error."""
    try:
        wav, _ = load(path, sr=16000, mono=True)
        return wav, len(wav)
    except Exception as e:
        return None, 0


def encode_audio_file(audio_path, codec, device):
    """Encode a single audio file using NeuCodec."""
    try:
        # Load audio at 16kHz (required by NeuCodec)
        wav, _ = load(audio_path, sr=16000, mono=True)
        
        # Convert to tensor format [1, 1, T]
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Encode to codes
        with torch.no_grad():
            codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        
        # Convert to list of integers for storage
        codes_list = codes.cpu().numpy().tolist()
        
        return codes_list
    except Exception as e:
        print(f"Error encoding {audio_path}: {e}")
        return None


def encode_audio_batch_parallel(audio_paths, codec, device, num_workers=4):
    """
    Load audio files in parallel, encode sequentially.
    NeuCodec doesn't support batch encoding, but parallel I/O speeds things up.
    
    Args:
        audio_paths: List of paths to audio files
        codec: NeuCodec model
        device: Device to use (passed to codec)
        num_workers: Number of threads for parallel audio loading
    
    Returns:
        List of codes for each audio
    """
    n = len(audio_paths)
    wavs = [None] * n
    
    # Parallel load all audio files
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_audio, path): i for i, path in enumerate(audio_paths)}
        for future in as_completed(futures):
            idx = futures[future]
            wav, _ = future.result()
            wavs[idx] = wav
    
    # Sequential encode (NeuCodec doesn't support batch)
    results = []
    for i, wav in enumerate(wavs):
        if wav is not None:
            try:
                wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
                results.append(codes.cpu().numpy().tolist())
            except Exception as e:
                results.append(None)
        else:
            results.append(None)
    
    return results


def prepare_dataset(metadata_path, audio_dir, output_path, device="cuda", batch_size=1, num_workers=4):
    """
    Prepare the Vietnamese dataset.
    
    Args:
        metadata_path: Path to metadata.csv
        audio_dir: Directory containing audio files
        output_path: Path to save the encoded dataset
        device: Device to use for encoding (cuda/cpu)
        batch_size: Number of audio files to encode in parallel
        num_workers: Number of threads for parallel audio loading
    """
    print("=" * 60)
    print("PREPARING VIETNAMESE DATASET FOR NEUTTS-AIR")
    print("=" * 60)
    
    # Load NeuCodec
    print(f"\n[1/4] Loading NeuCodec model on {device}...")
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec.eval().to(device)
    print("✓ NeuCodec loaded successfully!")
    
    # Read metadata (FAST - no file existence check)
    print(f"\n[2/4] Reading metadata from {metadata_path}...")
    dataset = []
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            audio_file = row['audio']
            transcript = row['transcript']
            audio_path = os.path.join(audio_dir, audio_file)
            
            dataset.append({
                'audio_file': audio_file,
                'audio_path': audio_path,
                'text': transcript
            })
    
    print(f"✓ Found {len(dataset)} samples")
    
    # Encode audio files (no sorting - avoids 600k+ getsize calls)
    print(f"\n[3/4] Encoding audio files with NeuCodec (batch_size={batch_size}, workers={num_workers})...")
    
    encoded_dataset = []
    error_count = 0
    
    if batch_size > 1:
        # Batch processing with parallel loading
        for i in tqdm(range(0, len(dataset), batch_size), desc="Encoding batches"):
            batch_samples = dataset[i:i + batch_size]
            batch_paths = [s['audio_path'] for s in batch_samples]
            
            codes_list = encode_audio_batch_parallel(batch_paths, codec, device, num_workers)
            
            for sample, codes in zip(batch_samples, codes_list):
                if codes is not None:
                    encoded_dataset.append({
                        'audio_file': sample['audio_file'],
                        'text': sample['text'],
                        'codes': codes
                    })
                else:
                    error_count += 1
    else:
        # Single file processing (original behavior)
        for sample in tqdm(dataset, desc="Encoding"):
            codes = encode_audio_file(sample['audio_path'], codec, device)
            
            if codes is not None:
                encoded_dataset.append({
                    'audio_file': sample['audio_file'],
                    'text': sample['text'],
                    'codes': codes
                })
            else:
                error_count += 1
    
    if error_count > 0:
        print(f"⚠️  Skipped {error_count} files due to errors")
    print(f"✓ Successfully encoded {len(encoded_dataset)} samples")
    
    # Save dataset
    print(f"\n[4/4] Saving encoded dataset to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(encoded_dataset, f)
    
    print(f"✓ Dataset saved successfully!")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(encoded_dataset)}")
    
    if encoded_dataset:
        avg_codes_len = sum(len(s['codes']) for s in encoded_dataset) / len(encoded_dataset)
        avg_text_len = sum(len(s['text']) for s in encoded_dataset) / len(encoded_dataset)
        print(f"Average codes length: {avg_codes_len:.1f}")
        print(f"Average text length: {avg_text_len:.1f} characters")
        
        print("\nSample data:")
        sample = encoded_dataset[0]
        print(f"  Audio: {sample['audio_file']}")
        print(f"  Text: {sample['text']}")
        print(f"  Codes length: {len(sample['codes'])}")
        print(f"  First 10 codes: {sample['codes'][:10]}")
    
    print("\n✅ Dataset preparation complete!")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Vietnamese dataset for NeuTTS-Air")
    parser.add_argument(
        "--metadata",
        type=str,
        default="metadata.csv",
        help="Path to metadata.csv file"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="wavs",
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vietnamese_dataset.pkl",
        help="Output path for encoded dataset"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for encoding"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of audio files to encode in parallel (default: 8)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of threads for parallel audio loading (default: 8)"
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        metadata_path=args.metadata,
        audio_dir=args.audio_dir,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
