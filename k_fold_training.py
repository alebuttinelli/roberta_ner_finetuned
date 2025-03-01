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
