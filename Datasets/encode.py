import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Define the data cache directory for encoded files
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "encoded")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

args = argparse.Namespace(model_desc="gpt-2")  # Simplified for illustration

# List of datasets to tokenize
datasets_to_tokenize = [
     ("HuggingFaceFW/fineweb", "sample-10BT", ["text"]), #hugging face repo, and "text" is the column to be extracted
     ("./SHL/data/", "", ['Sensor Data', 'Label']), # path to the local dataset, desried (here 2) columns to be extracted
     #("Path to the local/ huggingface dataset", "", ['column', 'names', 'to be extracted']),
]

CHUNK_SIZE = 100 * 1024 * 1024  # 100 MB in bytes

def tokenize_gpt2(doc):
    # Tokenizes a single document and returns a numpy array of uint16 tokens
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode_ordinary(s)
    eot = enc._special_tokens['<|endoftext|>']  # end of text token
    tokens = [eot]  # The special token delimits all documents

    combined_text = doc["text"]
    tokens.extend(encode(combined_text))
    tokens_np = np.array(tokens)

    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token dictionary too large for uint16"
    tokens_np_uint = tokens_np.astype(np.uint16)
    return tokens_np_uint

def process_dataset(dataset_name, dataset, output_dir):
    # Process each document and accumulate tokens until reaching CHUNK_SIZE
    token_buffer = []
    current_size = 0
    file_index = 0

    with mp.Pool(mp.cpu_count() - 2) as pool:
        for tokens in tqdm(pool.imap(tokenize_gpt2, dataset, chunksize=16), total=len(dataset), desc=f"Tokenizing {dataset_name}"):
            token_buffer.append(tokens)
            current_size += tokens.nbytes  # Add the size of the current token array

            # If accumulated size reaches or exceeds CHUNK_SIZE, save the chunk
            if current_size >= CHUNK_SIZE:
                save_chunk(output_dir, token_buffer, file_index)
                file_index += 1
                token_buffer = []  # Reset buffer
                current_size = 0   # Reset current size

    # Save any remaining tokens in the buffer
    if token_buffer:
        save_chunk(output_dir, token_buffer, file_index)

def save_chunk(output_dir, token_buffer, file_index):
    # Concatenate all tokens in the buffer and save them as a single file
    all_tokens_np = np.concatenate(token_buffer)
    filename = os.path.join(output_dir, f"chunk_{file_index:06d}.npy")
    np.save(filename, all_tokens_np)
    print(f"Saved {filename} with {len(all_tokens_np)} tokens")

def process_all_datasets():
    for dataset_name, remote_name, text_columns in datasets_to_tokenize:
        dataset = load_dataset(dataset_name, name=remote_name, split="train")

        def create_text(x):
            if 'Instruction' in text_columns:
                return "\n\n".join([f'### {col}:\n{x[col]}' for col in text_columns])
            else:
                return " ".join([x[col] for col in text_columns])

        dataset = dataset.map(lambda x: {"text": create_text(x)})

        # Prepare output directory
        output_dir = os.path.join(DATA_CACHE_DIR, dataset_name.replace("/", "").replace(".", ""))
        os.makedirs(output_dir, exist_ok=True)

        process_dataset(dataset_name, dataset, output_dir)

if __name__ == "__main__":
    process_all_datasets()

