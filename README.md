# MedSum - Clinical Note Summarization with BART and T5

This project fine-tunes two encoder–decoder Transformer models, **T5-small** and **BART-base**, to generate abstractive summaries of real clinical notes from the **MIMIC-IV-BHC** dataset.  
We compare the two models in terms of training dynamics and ROUGE-based summarization quality.

---

## Dataset

We use the **MIMIC-IV-BHC** dataset from PhysioNet.

- Contains **full clinical notes** written during hospitalization, along with **human-written discharge summaries** that we treat as ground truth.
- Size: roughly **300,000 rows** of high-quality clinical text.
- Due to the sensitive nature of the data, we had to:
  - Complete a certification process on **PhysioNet**.
  - Agree to strict usage and privacy requirements.

This dataset is ideal for our project because it provides long, realistic notes paired with concise expert summaries.

---

## Preprocessing

Our preprocessing pipeline is shared between **T5** and **BART** to ensure a fair comparison.

1. **Text Cleaning**
   - Remove PHI markers like `[** ... **]`.
   - Remove or normalize:
     - Newline characters.
     - Repeated punctuation (e.g., `---`, `===`).
     - Extra spaces and other minor formatting artifacts.

2. **Filter Low-Quality Examples**
   - Keep only notes with at least **1,300 characters**.
   - Keep only summaries with at least **300 characters**.
   - This removes extremely short or uninformative pairs and focuses training on richer examples.

3. **Train / Validation / Test Split**
   - Split the cleaned dataset into:
     - **81%** train
     - **9%** validation
     - **10%** test
   - The same split is used for both models.

---

## Tokenization

We use the native tokenizers from Hugging Face for each model.

### How Tokenization Works

- Text is broken into **tokens**, which are not always full words.
  - For example, `"pneumonia"` might be split into `"pneumon"` + `"ia"`.
  - Common words often stay whole, while rare medical terms may be split.
- Each token is then mapped to a **numeric ID** from the model’s vocabulary.
  - Example (T5):  
    - `"the"` → `3`  
    - `"patient"` → `1871`  
    - `"has"` → `65`

### Vocabulary Sizes

- **T5-small**: vocabulary of about **32,000** tokens.
- **BART-base**: vocabulary of about **50,000** tokens.

Both models therefore see the **same cleaned text**, but encode it using their own subword vocabularies and token IDs.

---

## T5 Model and Training

### Model

- **Model**: `t5-small` (~**60M parameters**).
- **Architecture**: encoder–decoder Transformer.
  - **Encoder**: processes the full input note and builds contextual representations.
  - **Decoder**: generates the summary **one token at a time**, attending to the encoder output.

### Training Objective

- **Loss function**: token-level **cross-entropy** between predicted next tokens and ground truth.
- During training, we use **teacher forcing**:
  - After each predicted token, the decoder is fed the **true** next token so it doesn’t drift off early with wrong words.

### Optimization Setup

- **Optimizer**: AdamW  
- **Learning rate**: `3e-5`  
- **Batch size**: `8`  
- **Epochs**: `3`  
- Training loss decreases sharply, especially in the **first two epochs**, which contribute most of the performance gains.

---

## BART Model and Training

### Model

- **Model**: `facebook/bart-base` (~**139M parameters**).
- **Architecture**: denoising encoder–decoder Transformer.
  - Pretrained to reconstruct clean text from **corrupted inputs**, which is well suited to summarization.
  - **Encoder**: processes the full clinical note.
  - **Decoder**: generates the summary token-by-token using **cross-attention** over the encoded note.

### Training Objective

- **Loss function**: same token-level **cross-entropy** as T5 (predicted vs. actual next tokens).

### Optimization Setup

- **Optimizer**: AdamW  
- **Learning rate**: `2e-5`  
- **Batch size**: `1` (safer for MacBook RAM with long sequences)  
- **Epochs**: **4**
- **Sequence lengths**:
  - **Input notes**: max **1024 tokens**
  - **Summaries**: max **512 tokens**

### Checkpointing

After each epoch:

- The Trainer saves a numbered checkpoint (`checkpoint-XXXX`).
- A small utility script copies the latest one to **`checkpoint-latest`**, making it easy to resume training or reload the best model later without losing previous epochs.

---

## Evaluation with ROUGE

We evaluate both models on the **held-out test set** using **beam search** at generation time:

- Beam search explores multiple possible summary continuations before choosing the final output.
- We then compute **ROUGE** scores **out of 100%** (we multiply the raw 0–1 scores by 100).

### Metrics

We use four ROUGE variants:

- **ROUGE-1**: unigram overlap (individual word matches).
- **ROUGE-2**: bigram overlap (consecutive word pairs).
- **ROUGE-L**: longest common subsequence between model summary and reference.
- **ROUGE-Lsum**: longest common subsequence computed over the **entire summary** text (segment-level ROUGE-L).

These scores give a quantitative measure of how much of the reference summary’s content the model captures.

---
## Repository Structure

A simplified view of the relevant files:

- `bart_data_cleaning.ipynb` –  
  Preprocessing and cleaning pipeline for MIMIC-IV-BHC (text cleaning, filtering, splitting, creation of `train_clean.csv`, `val_clean.csv`, `test_clean.csv`).

- `bart_training.ipynb` –  
  End-to-end BART workflow:
  - Load cleaned CSVs.
  - Convert to `datasets.Dataset`.
  - Tokenize body/summary pairs.
  - Define `DataCollatorForSeq2Seq`.
  - Configure `TrainingArguments`, train the model, and save checkpoints.
  - Evaluate on the test set and compute ROUGE metrics.
  - Generate example summaries for qualitative inspection.

- `train_t5.py` / `t5_tokenization.py` / `t5_evaluation.py` –  
  Scripts for T5 tokenization, training, and ROUGE evaluation using a similar pipeline to BART.

- `bart_base_mimic_checkpoints/`(not included) –  
  Directory containing Hugging Face checkpoints for BART, including `checkpoint-latest` for the final model.

- `train_clean.csv`, `val_clean.csv`, `test_clean.csv`(not included) –  
  Cleaned and filtered note/summary pairs used by both models.
