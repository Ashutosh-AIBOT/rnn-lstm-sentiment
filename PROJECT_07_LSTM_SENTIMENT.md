# Project 07 — Custom Neural Architecture: RNN/LSTM Sentiment Classification
**Level:** Intermediate | **Dataset:** IMDB (HuggingFace) | **Framework:** PyTorch (Custom Training from Scratch)

---

## Project Structure

```
07_lstm_sentiment/
├── notebooks/
│   ├── 01_eda.py               # EDA & text statistics
│   ├── 02_vocabulary.py        # Tokenization & vocab building
│   ├── 03_model.py             # Embedding, LSTM architecture & training
│   └── 04_pipeline.py          # Master runner — executes all stages end-to-end
├── data/
│   └── processed/
│       ├── vocab.pkl
│       └── processed_data.pkl
├── models/
│   └── model.pkl
├── dashboard_core.py
└── app.py
```

---

## Design Principles

- Every file is **self-contained and importable** — no circular dependencies
- All heavy data loading is **chunk-based** — never load full dataset into memory at once
- Every stage is **idempotent** — re-running saves over previous outputs cleanly
- All progress uses **tqdm** bars with descriptive postfixes
- Every STOP prints a clearly formatted **section header + explanation block**
- Functions are **small, named, and single-purpose** — no god functions
- `04_pipeline.py` is the single entry point that wires all three stages

---

## `01_eda.py` — Exploratory Data Analysis

### Purpose
Load the IMDB dataset, compute text statistics, and print findings with visual summaries. This file has no side effects — it only reads and reports.

### Functions to implement
- `load_dataset_split(split, batch_size)` — stream the HuggingFace dataset in chunks; yield `(texts, labels)` batches rather than loading all 25k rows at once
- `compute_length_stats(text_iter)` — accumulate word-count stats (mean, min, max, 95th percentile) using a running buffer without materialising the full list
- `plot_length_histogram(lengths, output_path)` — save a clean histogram using matplotlib; no display calls
- `compute_word_frequencies(text_iter)` — return a `Counter` built chunk by chunk
- `print_stop(stop_num, title, explanation)` — shared pretty-printer used across all stops; prints a box-bordered header, the explanation lines, and then waits for `input("Press Enter to continue...")` before returning

### Stops

**STOP 1 — Dataset overview**
Run `load_dataset_split` for both train and test splits in chunks. Print split sizes, 3 sample reviews (truncated to 120 chars), and labels. Then call `print_stop` with an explanation covering:
- What sentiment classification is and why it matters
- Why text requires different preprocessing than tabular data
- What a token is and why review length is highly variable
- Why variable length is a challenge for neural networks that expect fixed-size inputs

**STOP 2 — Length distribution**
Accumulate word counts across the train split chunk by chunk. Compute stats and save histogram. Print mean, min, max, and 95th-percentile length. Then call `print_stop` explaining:
- Why we choose the 95th percentile as `MAX_LEN` rather than the true max
- What padding and truncation do and when each is applied
- Why outlier lengths waste memory and slow training

**STOP 3 — Word frequency analysis**
Build the frequency `Counter` over train texts. Print the top 20 words and bottom 20 words. Then call `print_stop` explaining:
- Why very rare words (typos, names, one-offs) get dropped with a `MIN_FREQ` threshold
- Why stopwords are often kept for sentiment unlike in topic modelling
- What a vocabulary size of 10k–50k means in practice

---

## `02_vocabulary.py` — Tokenisation & Vocabulary

### Purpose
Build the vocabulary from scratch, encode all reviews to integer sequences, and persist both to disk. All processing is chunk-based.

### Constants (top of file)
```
MAX_VOCAB  = 25_000
MIN_FREQ   = 2
MAX_LEN    = (value from STOP 2 — passed in or loaded from stats file)
```

### Functions to implement
- `build_vocab(text_iter, max_vocab, min_freq)` — stream texts in chunks; accumulate a `Counter`; construct the final `{word: idx}` dict with `<PAD>=0` and `<UNK>=1` first; trim to `max_vocab` by frequency; return vocab dict
- `encode_text(text, vocab, max_len)` — lowercase, whitespace-split, map to indices (unknown → 1), truncate from the end to keep the conclusion, then right-pad with zeros to `max_len`; return a fixed-length list of ints
- `encode_split_chunked(split, vocab, max_len, chunk_size)` — stream a dataset split in `chunk_size` chunks; encode each chunk; yield `(encoded_batch, label_batch)` tuples; show a tqdm bar with postfix `encoded=N`
- `save_vocab(vocab, path)` and `load_vocab(path)` — thin wrappers around `pickle`
- `save_processed(data, path)` and `load_processed(path)` — same pattern for encoded sequences + labels

### Stops

**STOP 4 — Vocabulary construction**
Build the vocab from the train split chunk by chunk, showing a tqdm bar. Print final vocab size. Save to `data/processed/vocab.pkl`. Then call `print_stop` explaining:
- What `<PAD>` (index 0) is — the zero vector used to make short sequences reach `MAX_LEN`
- What `<UNK>` (index 1) is — the catch-all for out-of-vocabulary words at inference
- Why words are sorted by frequency before assigning indices
- The memory–coverage trade-off when choosing `MAX_VOCAB`

**STOP 5 — Text to integer sequences**
Encode the full train and test splits using `encode_split_chunked`, collecting the results. Show a tqdm bar per split. Print one example: original text → integer sequence (first 15 values shown). Save encoded data to `data/processed/processed_data.pkl`. Then call `print_stop` explaining:
- The full encode path: lowercase → split → map to ints → truncate from right → pad to `MAX_LEN`
- Why post-truncation (keeping the end of a review) preserves the conclusion where sentiment often lives
- What padding index 0 means and why it must not carry gradient signal

**STOP 6 — DataLoader inspection**
Instantiate `IMDBDataset` (defined in `03_model.py`) and two `DataLoader` objects (train and val). Print a single batch shape and label distribution. Then call `print_stop` explaining:
- Why all encoding is done before training rather than inside `__getitem__`
- The memory arithmetic: 25k reviews × 500 tokens × 4 bytes ≈ 50 MB — perfectly fine in RAM
- What `batch_first=True` means for subsequent LSTM layers

---

## `03_model.py` — Architecture, Training & Evaluation

### Purpose
Define the full model, train it, evaluate on the test set, and save the model checkpoint. All training state is logged and persisted for the dashboard.

### Classes and functions to implement

**`IMDBDataset(Dataset)`**
- `__init__(self, sequences, labels)` — store pre-encoded sequences and integer labels as tensors
- `__len__` and `__getitem__` — standard; `__getitem__` returns `(LongTensor[MAX_LEN], FloatTensor scalar)`

**`LSTMClassifier(nn.Module)`**
- `__init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout, pad_idx)`
  - `nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)`
  - `nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)`
  - `nn.Dropout(0.5)`
  - `nn.Linear(hidden_size, 1)`
  - `nn.Sigmoid()`
- `forward(self, x)` — embed → lstm → take `h_n[-1]` (last layer's final hidden state) → dropout → linear → sigmoid; return `[B, 1]`

**`train_one_epoch(model, loader, optimiser, criterion, device)`**
- Iterate loader with tqdm; accumulate loss and correct count; return `(avg_loss, accuracy)`

**`evaluate(model, loader, criterion, device)`**
- Same structure as train but with `torch.no_grad()`; also collect all predictions for F1 and confusion matrix; return `(avg_loss, accuracy, preds, targets)`

**`save_checkpoint(model, vocab, history, path)`** — save `state_dict`, vocab, and training history dict together

**`load_checkpoint(path, device)`** — return reconstructed model, vocab, and history

### Hyperparameters (top of file)
```
EMBED_DIM    = 128
HIDDEN_SIZE  = 256
NUM_LAYERS   = 2
DROPOUT      = 0.3
EPOCHS       = 15
LR           = 1e-3
BATCH_SIZE   = 64
MAX_NORM     = 1.0          # gradient clipping
```

### Stops

**STOP 7 — Embedding layer**
Instantiate only the embedding layer. Print weight shape. Run one random forward pass and print input/output shapes. Then call `print_stop` explaining:
- What an embedding layer physically is: a lookup table, not a matrix multiply
- Why `padding_idx=0` prevents gradient flow through PAD tokens
- What `embed_dim=128` means: each word becomes a point in 128-dimensional space
- How embeddings are learned jointly with the classifier and drift toward semantic similarity during training

**STOP 8 — LSTM architecture**
Instantiate the full `LSTMClassifier`. Print the model summary (parameter count per sub-module). Run one dummy batch and print all intermediate tensor shapes. Then call `print_stop` explaining:
- What `hidden_size=256` means: the dimension of `h_t` at each timestep
- What `num_layers=2` means: two stacked LSTM layers where the output of layer 1 feeds layer 2
- Why `h_n[-1]` is used instead of `out[:, -1, :]` (they are numerically identical for unidirectional LSTMs — use `h_n[-1]` as it is more explicit)
- The role of `h_n` (hidden state — short-term working memory) vs `c_n` (cell state — long-term memory highway)

**STOP 9 — LSTM cell mechanics**
Print a text diagram of the LSTM cell in markdown format. Then implement `lstm_cell_step(x, h_prev, c_prev, weights)` using raw matrix operations — no `nn.LSTM` here. Verify it produces the same output as a single-step `nn.LSTMCell`. Then call `print_stop` explaining:
- The forget gate: how much of `c_prev` to retain
- The input gate: how much new candidate information to write into the cell state
- Why `tanh` is applied to the cell state before computing `h_t` (bounds the output to `[-1, 1]`)
- Why the cell state's additive update (`f*c + i*g`) prevents the vanishing gradient problem that crippled plain RNNs
- The conceptual difference: RNN has only `h_t`; LSTM has both `h_t` and `c_t`

**STOP 10 — PackedSequence**
Show a 3-review mini-batch where reviews have lengths `[500, 312, 89]`. Demonstrate `pack_padded_sequence` → LSTM forward → `pad_packed_sequence`. Print the output shape before and after packing to illustrate that PAD steps are skipped. Then call `print_stop` explaining:
- What `PackedSequence` physically is: a flattened 1D buffer of only real tokens, plus a `batch_sizes` vector
- Why skipping PAD computation matters most when sequence length variance is high
- Why for IMDB (where we truncate everything to `MAX_LEN`) the speed gain is modest but the practice is still industry-standard

**STOP 11 — Training loop**
Run 15 epochs. For each epoch show a single tqdm bar covering all batches with a postfix showing live `loss` and `acc`. After each epoch print one summary line: `Epoch N/15 | train loss X.XXX acc X.XX% | val loss X.XXX acc X.XX%`. Apply gradient clipping before `optimiser.step()`. Save full history to a JSON file after every epoch (not just at the end) so the dashboard can read partial results. Then call `print_stop` explaining:
- Why gradient clipping is essential for RNNs: BPTT unrolls the graph across timesteps, and multiplied Jacobians compound into exploding norms
- What `max_norm=1.0` does mechanically: if the global gradient norm exceeds 1, each gradient is scaled by `1/norm` so the total norm equals 1
- How `BCELoss` + `Sigmoid` differs from `CrossEntropyLoss` + `Softmax` for binary classification

**STOP 12 — Evaluation & embedding visualisation**
Run inference over the test split. Print accuracy, F1 (macro), and a formatted confusion matrix. Extract `model.embedding.weight.data` and run PCA to 2D over the top 200 most frequent words. Save the 2D coordinates and word labels to a JSON file. Then call `print_stop` explaining:
- What embedding visualisation reveals: clusters of semantically related words emerge after training even though no explicit semantic signal was provided
- How the learned geometry differs from random initialisation

**STOP 13 — Save & inference demo**
Save the full checkpoint with `save_checkpoint`. Write `predict_sentiment(text, model, vocab, max_len, device)` and run it on 3 hard-coded reviews. Print `text → Positive (0.87)` style output for each. Then call `print_stop` explaining:
- The full inference chain: raw string → lowercase/split → map to vocab (unknown → UNK) → truncate from right → pad to `MAX_LEN` → `LongTensor([1, MAX_LEN])` → model → `.item()` → threshold at 0.5
- Why the vocab saved at training time must be used at inference — a different vocab produces wrong indices
- What happens at inference when the user's review contains words not in the training vocab

---

## `04_pipeline.py` — Master Runner

### Purpose
Single entry point that imports and calls the three stage files in sequence. Running `python notebooks/04_pipeline.py` executes the entire project from raw data to saved model with all stops and progress bars intact.

### Structure
- Import stage runner functions from `01_eda`, `02_vocabulary`, and `03_model`
- Define `run_pipeline()` with clear section separators printed to stdout
- Each stage is wrapped in a `try/except` so a failure in one stage prints a clear error and exits with a non-zero code rather than a silent crash
- At the very top, print a summary table of all 13 stops and which stage they belong to
- At the very end, print a completion summary: total time elapsed, files written, model path, and a one-line command to launch the Streamlit app

### Progress contract
Every stage calls its own tqdm bars internally. `04_pipeline.py` only prints stage-level separators and the final summary. It does not duplicate inner progress output.

### Entry-point guard
```
if __name__ == "__main__":
    run_pipeline()
```

---

## `dashboard_core.py` — Dashboard Utilities

### Purpose
Clean interface between the trained artefacts and the Streamlit UI. No training logic here — only loading and transformation.

### Functions
- `load_model_and_vocab(checkpoint_path, device)` → `(LSTMClassifier, vocab_dict)`
- `predict_sentiment(text, model, vocab, max_len, device)` → `{"label": str, "probability": float}`
- `get_training_curves(history_path)` → `{"epochs": list, "train_loss": list, "val_loss": list, "train_acc": list, "val_acc": list}`
- `get_embedding_viz(checkpoint_path, top_n)` → `{"words": list[str], "x": list[float], "y": list[float]}`
- `get_confusion_matrix(checkpoint_path)` → `{"matrix": list[list[int]], "labels": ["Negative", "Positive"]}`

---

## `app.py` — Streamlit Dashboard

### Layout (≈ 80 lines)

The app uses a single-page layout with a sidebar for model config and a main panel with tabs.

**Sidebar**
- Model checkpoint path selector (default to `models/model.pkl`)
- Device selector: CPU / CUDA / MPS
- One-line model status indicator: loaded / not found

**Main panel — top section**
- Large text area labelled "Write or paste a movie review"
- "Analyse Sentiment" button
- On click: show a coloured badge (`🟢 Positive` or `🔴 Negative`) and a horizontal probability gauge bar rendered with `st.progress`

**Main panel — tabs**

*Tab 1: Training curves*
- Two side-by-side line charts: loss curves (train + val) and accuracy curves (train + val)
- Use `st.line_chart` or plotly; show epoch on x-axis
- Below charts: three metric cards — final val accuracy, val loss, total epochs

*Tab 2: Embedding space*
- Scatter plot of the top 200 words projected to 2D via PCA
- Words labelled directly on the chart; sentiment-bearing words highlighted in a distinct colour
- Brief caption explaining what proximity in the space means

*Tab 3: Confusion matrix*
- Heatmap of the 2×2 confusion matrix with cell counts and row-normalised percentages
- Below: F1 score (macro) displayed as a metric

### UI standards
- `st.set_page_config(layout="wide", page_title="LSTM Sentiment")`
- All heavy loading (`load_model_and_vocab`, `get_training_curves`) wrapped in `@st.cache_resource` or `@st.cache_data`
- Spinner shown during prediction: `with st.spinner("Analysing...")`
- Error messages use `st.error` rather than bare `print` or `raise`
- No raw dataframes shown to the user — all data rendered as charts or styled tables

---

## Key Concepts Covered

All 13 stops together teach the following in sequence:

| Stop | Concept |
|------|---------|
| 1–3  | Text as data: tokens, variable length, frequency distributions |
| 4–5  | Vocabulary mechanics: `<PAD>`, `<UNK>`, index mapping |
| 6    | PyTorch Dataset / DataLoader contract |
| 7    | Embedding layer as a trainable lookup table |
| 8    | LSTM architecture: layers, hidden size, `h_n` vs `out` |
| 9    | LSTM gates from first principles: forget, input, cell, output |
| 10   | PackedSequence and variable-length sequence handling |
| 11   | Training loop: gradient clipping, BCELoss, epoch logging |
| 12   | Evaluation: F1, confusion matrix, embedding PCA visualisation |
| 13   | Inference pipeline and the importance of vocab consistency |
