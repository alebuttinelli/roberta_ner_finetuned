#carica il dataset di btc dall'hub di huggingface
dataset = load_dataset('strombergnlp/broad_twitter_corpus')

#However, this adds some special tokens [CLS] and [SEP] and the subword tokenization
#creates a mismatch between the input and labels. A single word corresponding to a single
#label may now be split into two subwords. You’ll need to realign the tokens and labels by:
#-Mapping all tokens to their corresponding word with the word_ids method.
#-Assigning the label -100 to the special tokens [CLS] and [SEP] so they’re ignored by the PyTorch loss function (see CrossEntropyLoss).
#-Only labeling the first token of a given word. Assign -100 to other subtokens from the same word.

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

#applicare con map la funzione di tokenizzazione e allineamento a tutto il dataset
tokenized_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True,
)

#data collation per il padding
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

#Seqeval actually produces several scores: precision, recall, F1, and accuracy.
seqeval = evaluate.load("seqeval")

#They should be set by two dictionaries, id2label and label2id,
#which contain the mappings from ID to label and vice versa

id2label = {i: label for i, label in enumerate(label_list)}
label2id = {v: k for k, v in id2label.items()}

#specifica la configurazione da XLMRoberta-large
config_roberta = XLMRobertaConfig.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
#config_bert = BertConfig.from_pretrained('bert-base-cased')

model = RoBERTaforNER(config=config_roberta, num_labels=len(label_list))

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

#dopo aver trovato i migliori iperparametri sul modello e sul dataset
#definire il training sul dataset con ten-fold cross validation per il training definitivo
#studia meglio la documentazione di KFold per capire cosa produce come output
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

from sklearn.model_selection import KFold

#crea oggetto kfold
k_folds = KFold(n_splits=10, random_state=27, shuffle=True)
#crea dataset continuo a partire da btc


for i, (train_index, test_index) in k_folds.split(dataset_tot):
  #crea dataset per ogni iterazione da sottoporre a training
  fold_dataset = DatasetDict({
    "train": dataset_tot.select(train_index),
    "test": dataset_tot.select(test_index)})
  #train dell'iterazione
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=fold_dataset["train"],
    eval_dataset=fold_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics)

  trainer.train()
