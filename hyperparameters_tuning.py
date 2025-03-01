#informarsi meglio se e come è possibile applicare diversi metodi di ricerca (grid search, random search, ecc.) con Trainer

#definisci lo spazio di ricerca degli iperparametri
#trial per variabili:
#optuna.trial.Trial.suggest_categorical() for categorical parameters
#optuna.trial.Trial.suggest_int() for integer parameters
#optuna.trial.Trial.suggest_float() for floating point parameters

#altri iperparametri (gli iperparametri sono quelli presenti in TrainingArguments):
#dimensioni batch ("per_device_train_batch_size")
#num train epoche  ("num_train_epochs")
#ottimizzatore ("optim" (str or training_args.OptimizerNames, optional, defaults to "adamw_torch") — The optimizer to use: adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused, adamw_anyprecision or adafactor.)
#learning rate ("learning_rate" (float, optional, defaults to 5e-5) — The initial learning rate for AdamW optimizer.)
#learning rate scheduler ("lr_scheduler_type" (str or SchedulerType, optional, defaults to "linear") — The scheduler type to use. See the documentation of SchedulerType for all possible values.)
#warmup ratio ("warmup_ratio" (float, optional, defaults to 0.0) — Ratio of total training steps used for a linear warmup from 0 to learning_rate.)
#weight decay ("weight_decay" (float, optional, defaults to 0) — The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer.)
#momentum
#diversa inizializzazione dei pesi (forse è "seed" come argomento di TrainingArguments)

def optuna_hp_space(trial):
    return {
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [3, 4, 6, 8, 10]),
        "optim": trial.suggest_categorical("optim", ["adamw_hf",
                                                     "adamw_torch",
                                                     "adamw_torch_fused",
                                                     "adamw_apex_fused",
                                                     "adamw_anyprecision",
                                                     "adafactor"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ['linear',
                                                                             'cosine',
                                                                             'cosine_with_restarts',
                                                                             'polynomial',
                                                                             'constant',
                                                                             'constant_with_warmup',
                                                                             'inverse_sqrt',
                                                                             'reduce_lr_on_plateau']),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1, log=True),
        "warmup_ratio": trail.suggest_float("warmup_ratio", 0.0, 0.1),
        "seed": trail.suggest_categorical("seed", [21, 42, 84])
    }


#definisci la funzione init del modello da testare per gli iperparametri
def model_init():
    return AutoModelForTokenClassification.from_pretrained(
        "xlm-roberta-large-finetuned-conll03-english", return_dict=True)

#istanzia Trainer per test iperparametri
trainer_iper = Trainer(
    model=None,
    args=training_args,
    train_dataset=tokenized_btc["train"],
    eval_dataset=tokenized_btc["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    model_init=model_init,
    data_collator=data_collator,
)

best_trial = trainer_iper.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=10,
)
