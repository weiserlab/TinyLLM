import os
import argparse
import numpy as np
from tqdm import tqdm
from data_common import write_datafile
import random

# Argument parser for input ratios and output directory
parser = argparse.ArgumentParser(description="Merge datasets by reading content proportionally and saving fixed-size shards.")
parser.add_argument("-d1", "--dataset1_ratio", type=float, default=0.3, help="Ratio of content to take from SHL dataset (default: 0.3)")
parser.add_argument("-d2", "--dataset2_ratio", type=float, default=0.7, help="Ratio of content to take from Fineweb dataset (default: 0.7)")
parser.add_argument("-o", "--output_dir", type=str, default="merged_output", help="Output directory to save merged shards (default: 'merged_output')")

args = parser.parse_args()

# Configuration
dataset1_path = os.path.join(os.path.dirname(__file__), "encoded", "SHL")
dataset2_path = os.path.join(os.path.dirname(__file__), "encoded", "Fineweb")
TOTAL_TOKENS = 9_000_000_000  # 9 billion tokens
TRAIN_RATIO = 0.98  # 98% for training
VALIDATION_RATIO = 1 - TRAIN_RATIO  # 2% for validation
SHARD_SIZE = 100 * 1024 * 1024  # 100 MB per shard, adjust according to your needs

def load_tokens_lazy(dataset_path):
    """Generator that yields token arrays from a dataset folder one by one."""
    files = sorted([f for f in os.listdir(dataset_path) if f.endswith(".npy")])
    for file in files:
        yield np.load(os.path.join(dataset_path, file))

def count_total_tokens(dataset_path):
    """Calculate the total number of tokens available in a dataset directory."""
    total_tokens = 0
    for tokens in load_tokens_lazy(dataset_path):
        total_tokens += len(tokens)
    return total_tokens

def merge_and_save_tokens_lazy(dataset1_path, dataset2_path, ratio1, ratio2, output_dir, shard_prefix, max_tokens):
    """Merge tokens from datasets with given ratios and save them in fixed-size shards."""
    os.makedirs(output_dir, exist_ok=True)
    total_token_count = 0  # Tracks the total number of tokens processed
    current_shard_token_count = 0  # Tracks the number of tokens in the current shard
    shard_index = 0
    all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)
    progress_bar = None

    # Initialize generators
    dataset1_gen = load_tokens_lazy(dataset1_path)
    dataset2_gen = load_tokens_lazy(dataset2_path)

    # Flags to track if datasets are exhausted
    dataset1_exhausted = False
    dataset2_exhausted = False

    # Probability calculation for selecting from either dataset
    prob_dataset1 = ratio1 / (ratio1 + ratio2)
    prob_dataset2 = ratio2 / (ratio1 + ratio2)

    while total_token_count < max_tokens:
        print(f'Current total token count: {total_token_count}, Max tokens: {max_tokens}')

        # Check if both datasets are exhausted
        if dataset1_exhausted and dataset2_exhausted:
            print("Both datasets are exhausted.")
            break

        # Randomly choose from dataset1 or dataset2 based on the specified probabilities
        if not dataset1_exhausted and random.random() < prob_dataset1:
            try:
                tokens = next(dataset1_gen)
            except StopIteration:
                dataset1_exhausted = True
                prob_dataset1 = 0
                prob_dataset2 = 1
                continue
        elif not dataset2_exhausted:
            try:
                tokens = next(dataset2_gen)
            except StopIteration:
                dataset2_exhausted = True
                prob_dataset1 = 1
                prob_dataset2 = 0
                continue
        else:
            # Skip this iteration if both datasets are exhausted
            continue

        # Check shard size and write out if necessary
        if current_shard_token_count + len(tokens) <= SHARD_SIZE:
            all_tokens_np[current_shard_token_count:current_shard_token_count + len(tokens)] = tokens
            current_shard_token_count += len(tokens)
            total_token_count += len(tokens)  # Increment total token count
            if progress_bar is None:
                progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"{shard_prefix} Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # Write the shard and reset
            filename = os.path.join(output_dir, f"{shard_prefix}_shard_{shard_index:06d}.bin")
            remainder = SHARD_SIZE - current_shard_token_count
            all_tokens_np[current_shard_token_count:current_shard_token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np[:SHARD_SIZE].tolist(), "gpt-2")
            shard_index += 1
            total_token_count += remainder  # Increment total token count
            current_shard_token_count = len(tokens) - remainder
            all_tokens_np[0:current_shard_token_count] = tokens[remainder:]
            progress_bar = None
            print(f"Saved shard {shard_index} with {SHARD_SIZE} tokens.")

    # Save any remaining tokens
    '''
    if current_shard_token_count != 0:
        filename = os.path.join(output_dir, f"{shard_prefix}_shard_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:current_shard_token_count].tolist(), "gpt-2")
        print(f"Saved final shard {shard_index} with {current_shard_token_count} tokens.")
        '''

def main():
    global TOTAL_TOKENS 
    # Calculate the number of tokens for training and validation based on available tokens
    train_tokens = int(TOTAL_TOKENS * TRAIN_RATIO)
    val_tokens = TOTAL_TOKENS - train_tokens  # Ensure all tokens are used
    print(f"Training tokens: {train_tokens}, Validation tokens: {val_tokens}")

    # Merge and save for training set
    merge_and_save_tokens_lazy(dataset1_path, dataset2_path, args.dataset1_ratio, args.dataset2_ratio, args.output_dir, "train", train_tokens)

    # Merge and save for validation set
    merge_and_save_tokens_lazy(dataset1_path, dataset2_path, args.dataset1_ratio, args.dataset2_ratio, args.output_dir, "val", val_tokens)

if __name__ == "__main__":
    main()

