import torch
import time
import sys
import os
from notebooks.s01_eda import run_eda
from notebooks.s02_vocabulary import run_vocab
from notebooks.s03_model import run_training

def print_summary_table():
    print("\n" + "█"*60)
    print("  PROJECT 07 — SENTIMENT LSTM PIPELINE v2.0")
    print("█"*60)
    print("\n13 STOPS ARCHITECTURAL MAP:")
    stops = [
        "1  Dataset overview (Train/Test splits)",
        "2  Length distribution (95th Percentile)",
        "3  Word frequency analysis (MIN_FREQ)",
        "4  Vocabulary construction (<PAD>, <UNK>)",
        "5  Text to integer sequences (Encoding)",
        "6  DataLoader inspection (Batching)",
        "7  Embedding layer (Trainable lookup table)",
        "8  LSTM architecture (Stacking & Memory)",
        "9  LSTM cell mechanics (Gates simulation)",
        "10 PackedSequence (Performance optimization)",
        "11 Training loop (Gradient Clipping)",
        "12 PCA Analysis (Embedding geometry)",
        "13 Inference demo (End-to-end chain)"
    ]
    for s in stops:
        print(f"  [ ] STOP {s}")
    print("█"*60 + "\n")

def run_pipeline():
    """Master runner for the 13-STOP Sentiment LSTM project."""
    auto_mode = "--auto" in sys.argv
    print_summary_table()
    
    if auto_mode:
        print(" [SYSTEM] MODE: AUTOMATED (NON-INTERACTIVE)")
    else:
        print(" [SYSTEM] MODE: INTERACTIVE (EDUCATIONAL)")
        
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" [SYSTEM] Found device: {device.upper()}")
    
    try:
        # STAGE 1: Exploratory Data Analysis (STOPS 1-3)
        print("\n>>> STAGE 1: DATA EXPLORATION")
        max_len = run_eda(auto_mode)
        
        # STAGE 2: Vocabulary & Encoding (STOPS 4-6)
        print("\n>>> STAGE 2: LINGUISTIC PREPROCESSING")
        vocab = run_vocab(max_len, auto_mode)
        
        # STAGE 3: Model, Packing & Training (STOPS 7-13)
        print("\n>>> STAGE 3: NEURAL TRAINING & VISUALIZATION")
        model = run_training(vocab, device, auto_mode)
        
        duration = time.time() - start_time
        print(f"\n" + "═"*60)
        print(f"  [COMPLETE] Total Pipeline Time: {duration/60:.2f} mins")
        print("═"*60)
        print("  All 13 Stops successfully verified.")
        print("  Dashboard is now ready to visualize metrics.")
        print("═"*60)
        print("\nLaunch Dashboard with: streamlit run app.py")
        
    except KeyboardInterrupt:
        print("\n" + "!"*60)
        print("  [ABORT] Pipeline stopped by user.")
        print("!"*60)
    except Exception as e:
        print("\n" + "!"*60)
        print(f"  [ERROR] Pipeline failed: {str(e)}")
        print("!"*60)
        raise e

if __name__ == "__main__":
    run_pipeline()
