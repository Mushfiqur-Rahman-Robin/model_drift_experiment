# BERT Model

## Overview
The BERT model wrapper utilizes the `bert-base-uncased` model from HuggingFace Transformers. It is designed to handle text data and extract rich semantic embeddings.

## Usage
- **Input**: List of text strings.
- **Tokenizer**: Standard `BertTokenizer`.
- **Model**: Pre-trained `BertModel`.

## Drift Detection Features
We use the **CLS token embedding** (768 dimensions) from the last hidden state as the feature vector for drift detection. This provides a summary representation of the input sentence, allowing us to detect semantic shifts (e.g., vocabulary changes, topic shifts).

## Caching
The model weights and tokenizer are cached locally in the `models/bert` directory to ensure reproducibility and offline capability.
