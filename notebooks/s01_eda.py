import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
import time

# --- SHARED UTILS ---
def print_stop(stop_num, title, explanation, auto_mode=False):
    """Shared pretty-printer for pipeline segments."""
    print("\n" + "═"*60)
    print(f"  STOP {stop_num} — {title.upper()}")
    print("═"*60)
    print(f"\nEXPLANATION:\n{explanation}")
    print("─"*60)
    if not auto_mode:
        input("\n[Action] Press Enter to continue and execute this stop...")
    else:
        print(f"\n[Auto Mode] Proceeding from STOP {stop_num}...")
        time.sleep(1) # Visual pause

def load_data_chunked(split="train", batch_size=1000):
    """Stream HuggingFace IMDB dataset in chunks."""
    dataset = load_dataset("imdb", split=split, streaming=True)
    batch_texts, batch_labels = [], []
    for entry in dataset:
        batch_texts.append(entry['text'])
        batch_labels.append(entry['label'])
        if len(batch_texts) == batch_size:
            yield batch_texts, batch_labels
            batch_texts, batch_labels = [], []
    if batch_texts:
        yield batch_texts, batch_labels

# --- STOPS ---
def stop_1_overview(auto_mode=False):
    print("\n[STOP 1] Analyzing Dataset Splits...")
    train_gen = load_data_chunked("train", 500)
    texts, labels = next(train_gen)
    
    print(f"Sample size: {len(texts)} reviews loaded (Initial Chunk)")
    print(f"Label distribution (Chunk 1): {Counter(labels)}")
    print("\nSample Review (Truncated):")
    print(f"\"{texts[0][:150]}...\" -> Label: {'Positive' if labels[0] == 1 else 'Negative'}")
    
    explanation = (
        "• Sentiment classification determines the emotive polarity of text (Positive vs Negative).\n"
        "• Unlike tabular data, text is unstructured and highly variable in length.\n"
        "• Tokenization is required to turn raw strings into numerical indices for the model.\n"
        "• Variable sequence lengths are a primary challenge for LSTMs that expect batched tensors."
    )
    print_stop(1, "Dataset Overview", explanation, auto_mode)

def stop_2_length_analysis(auto_mode=False):
    print("\n[STOP 2] Computing Sequence Lengths...")
    lengths = []
    # Total IMDB train is 25000
    for texts, _ in tqdm(load_data_chunked("train", 2000), total=13, desc="Processing lengths"):
        lengths.extend([len(t.split()) for t in texts])
    
    mean_len = np.mean(lengths)
    max_len_final = int(np.percentile(lengths, 95))
    
    print(f"\nMean Word Count: {mean_len:.1f}")
    print(f"95th Percentile (Target MAX_LEN): {max_len_final}")
    
    # Plotting
    os.makedirs("charts", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=50, color='#7F77DD', alpha=0.7)
    plt.axvline(max_len_final, color='red', linestyle='--', label=f'95th Percentile ({max_len_final})')
    plt.title("IMDB Review Length Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("charts/length_dist.png")
    print("Chart saved to charts/length_dist.png")
    
    explanation = (
        "• We choose the 95th percentile as MAX_LEN to balance coverage vs memory efficiency.\n"
        "• Truncation: Reviews longer than MAX_LEN lose data (usually keeping the conclusion).\n"
        "• Padding: Reviews shorter than MAX_LEN are zero-filled to maintain uniform batch shapes.\n"
        "• Extremely long outliers waste GPU memory and slow down training significantly."
    )
    print_stop(2, "Length Distribution", explanation, auto_mode)
    return max_len_final

def stop_3_word_frequency(auto_mode=False):
    print("\n[STOP 3] Analyzing Vocabulary Frequency...")
    freqs = Counter()
    for texts, _ in tqdm(load_data_chunked("train", 2000), total=13, desc="Counting words"):
        for t in texts:
            freqs.update(t.lower().split())
    
    print("\nTop 20 Words:", [w for w, c in freqs.most_common(20)])
    print("Rare Words (tail):", [w for w, c in list(freqs.items())[-10:]])
    
    explanation = (
        "• Rare words (count < MIN_FREQ) are replaced with <UNK> to keep vocab manageable.\n"
        "• IMDB contains high-freq stopwords ('the', 'and') which often help sentiment context.\n"
        "• A typical production vocabulary size for this task is between 10k and 25k words.\n"
        "• Filtering out the 'long tail' reduces noise and prevents overfitting on typos."
    )
    print_stop(3, "Word Frequency", explanation, auto_mode)

def run_eda(auto_mode=False):
    stop_1_overview(auto_mode)
    max_len = stop_2_length_analysis(auto_mode)
    stop_3_word_frequency(auto_mode)
    return max_len

if __name__ == "__main__":
    run_eda()
