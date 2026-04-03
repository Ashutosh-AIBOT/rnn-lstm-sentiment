import os
import pickle
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset
import torch
import time
from notebooks.s01_eda import load_data_chunked, print_stop

# --- CONSTANTS ---
MAX_VOCAB = 25000
MIN_FREQ = 2

def build_vocab(text_iter, max_vocab, min_freq):
    """Build vocabulary from texts."""
    freqs = Counter()
    # Total train is 25000 / 2000 ~= 13 chunks
    for texts, _ in tqdm(text_iter, total=13, desc="Building vocab"):
        for t in texts:
            freqs.update(t.lower().split())
    
    # Filter by frequency and trim to max_vocab
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, count in freqs.most_common(max_vocab):
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

def encode_text(text, vocab, max_len):
    """Map string to fixed-length integer list."""
    words = text.lower().split()
    # Unknown -> 1
    encoded = [vocab.get(w, 1) for w in words]
    # Truncate and Pad
    encoded = encoded[:max_len]
    pad_len = max_len - len(encoded)
    return encoded + [0] * pad_len

def encode_and_save(split, vocab, max_len, path):
    """Process a split and save to pickle."""
    data = []
    # Train is 25k, Test is 25k. 25000 / 2000 ~= 13 chunks
    for texts, labels in tqdm(load_data_chunked(split, 2000), total=13, desc=f"Encoding {split}"):
        for t, l in zip(texts, labels):
            data.append((encode_text(t, vocab, max_len), l))
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {split} data to {path} ({len(data)} items)")

# --- STOPS ---
def stop_4_vocab(auto_mode=False):
    print("\n[STOP 4] Building Vocabulary...")
    vocab = build_vocab(load_data_chunked("train", 2000), MAX_VOCAB, MIN_FREQ)
    print(f"Final Vocab Size: {len(vocab)} words")
    
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    
    explanation = (
        "• <PAD> (index 0) prevents gradient signaling through padding tokens.\n"
        "• <UNK> (index 1) handles words never seen during training or filtered by MIN_FREQ.\n"
        "• Trimming to 25k words ensures the embedding matrix size remains efficient.\n"
        "• Words are sorted by frequency to prioritize the most semantically relevant tokens."
    )
    print_stop(4, "Vocab Construction", explanation, auto_mode)
    return vocab

def stop_5_encoding(vocab, max_len, auto_mode=False):
    print("\n[STOP 5] Encoding Sequences...")
    sample_text = "The movie was absolutely amazing and highly recommended."
    encoded = encode_text(sample_text, vocab, max_len)
    
    print(f"Original: \"{sample_text}\"")
    print(f"Encoded (first 15): {encoded[:15]}")
    
    # Save both splits
    encode_and_save("train", vocab, max_len, "data/processed/train_data.pkl")
    encode_and_save("test", vocab, max_len, "data/processed/test_data.pkl")
    
    explanation = (
        "• Sequence encoding converts qualitative text into static integer vectors.\n"
        "• Right-padding ensures all input tensors in a batch have unified shapes [B, MAX_LEN].\n"
        "• Truncation from the end (right) is preferred in reviews as sentiment often peaks in the conclusion."
    )
    print_stop(5, "Text Encoding", explanation, auto_mode)

def stop_6_dataloader_check(auto_mode=False):
    explanation = (
        "• DataLoaders handle dataset shuffling, batching, and parallelized loading.\n"
        "• Pre-encoding data off-line drastically reduces training-time compute overhead.\n"
        "• The resulting tensor shape [BATCH_SIZE, MAX_LEN] is the direct input for the Embedding layer."
    )
    print_stop(6, "Dataset Verification", explanation, auto_mode)

def run_vocab(max_len, auto_mode=False):
    vocab = stop_4_vocab(auto_mode)
    stop_5_encoding(vocab, max_len, auto_mode)
    stop_6_dataloader_check(auto_mode)
    return vocab

if __name__ == "__main__":
    run_vocab(256)
