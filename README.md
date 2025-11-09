# Custom RoBERTa-BiLSTM for Named Entity Recognition (NER)

This repository contains the code for a custom, high-performance Named Entity Recognition (NER) model. It combines the deep contextual understanding of a pre-trained Transformer (like **XLM-RoBERTa**) with the sequential pattern-matching strength of a **Bidirectional LSTM (BiLSTM)** layer.

This architecture is designed to improve upon the standard `AutoModelForTokenClassification` by adding a recurrent layer that can explicitly capture long-range token dependencies, which is often beneficial for NER tasks.

## Model Architecture

The custom model (`RoBERTaforNER` in `model.py`) is a PyTorch module that consists of the following components, processed in order:

1.  **RoBERTa Base Model:** We load a pre-trained Transformer model (e.g., `xlm-roberta-large`) *without* its classification head (`AutoModel`). This acts as our powerful text embedder.
2.  **Dropout:** A standard dropout layer is applied to the RoBERTa's `last_hidden_state` for regularization.
3.  **Optional BiLSTM Layer:** (Default: **Enabled**) The `last_hidden_state` from RoBERTa is fed into a multi-layer Bidirectional LSTM. This allows the model to re-process the contextual embeddings, reading the entire sequence from left-to-right and right-to-left to build an even richer understanding of token relationships.
4.  **Classifier Head:** A final `nn.Linear` layer maps the output of the BiLSTM (or the RoBERTa, if the BiLSTM is disabled) to the number of NER labels (e.g., 9 labels for `conll2003`).

## Setup and installation

1. **Clone the respository**:
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

2. **Create a virtual environment**:
```bash
python -m venv venv
```

Activate it:
macOS/Linux: source venv/bin/activate
Windows: venv\Scripts\activate

3. **Install the dependencies: This will install transformers, datasets, evaluate, seqeval, torch, and accelerate.**
```bash
pip install -r requirements.txt
```

## How to train
All training is handled by the train.py script, which uses the Hugging Face Trainer API for a robust and efficient training loop.

You can configure the model, dataset, and training parameters using command-line arguments.

**Example 1: Training with BiLSTM (Recommended)**
This command trains the custom model on the conll2003 dataset using the BiLSTM layer.
```bash
python train.py \
    --model_name "xlm-roberta-large" \
    --dataset_name "conll2003" \
    --output_dir "./models/roberta-bilstm-conll" \
    --num_epochs 3 \
    --batch_size 8 \
    --use_bilstm
```

**Example 2: Training without BiLSTM (Baseline)**
By simply omitting the --use_bilstm flag, you can train a baseline model (just RoBERTa + Classifier) to compare performance.
```bash
python train.py \
    --model_name "xlm-roberta-large" \
    --dataset_name "conll2003" \
    --output_dir "./models/roberta-baseline-conll" \
    --num_epochs 3 \
    --batch_size 8
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
