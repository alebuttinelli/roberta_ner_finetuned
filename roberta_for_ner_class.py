pip install transformers datasets seqeval evaluate accelerate optuna
from transformers import AutoTokenizer, BertPreTrainedModel, XLMRobertaForTokenClassification, XLMRobertaModel, XLMRobertaForMaskedLM, DataCollatorForTokenClassification, get_scheduler, AutoModelForTokenClassification, XLMRobertaTokenizerFast,TrainingArguments, Trainer, XLMRobertaConfig
from transformers import AutoModel, BertConfig
from datasets import load_dataset, DatasetDict
import torch
import evaluate
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch.nn as nn

class RoBERTaforNER(BertPreTrainedModel):
  #num_layers_lstm indica il numero di layers lstm impilati uno sopra l'altro (non abbiamo quindi bisogno di scrivere per due volte il livello di bilstm, basta mettere a 2 il parametro)
    #hidden_size_lstm indica il numero di neuroni nei livelli della lstm. Bisogna impostarlo a 768 come la lunghezza degli embed di BERT
    #questi nelle parentesi sono i parametri che possiamo specificare nella classe
    #num_labels non è predefinito, dipende dal dataset con cui addestriamo il modello
    #labels anche se compare solo come parametro di forward() può essere specificato come parametro al momento di usare il modello su istanze specifiche
    #se si specifica le istanze vengono allenate, se no si predice il loro tag
    def __init__(self, config, num_labels, hidden_size_lstm=1024, num_layers_lstm=2):
        super().__init__(config)
        self.num_labels = num_labels
        #self.config = XLMRobertaConfig.from_pretrained('xlm-roberta-large')
        #self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-large-finetuned-conll03-english", config=self.config)
        self.roberta = AutoModel.from_pretrained("xlm-roberta-large-finetuned-conll03-english", config=self.config)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        #due livelli di bilstm
        #self.bilstm = nn.LSTM(input_size=self.config.hidden_size, hidden_size=hidden_size_lstm, num_layers=num_layers_lstm, bidirectional=True, batch_first=True)
        #hidden_size_lstm è moltiplicata per 2 in quanto riceve come input l'embed lstm in entrambe le direzioni (vettori concatenati)
        self.classifier = nn.Linear(hidden_size_lstm, self.num_labels)

        self.init_weights()

    #se labels=None (default) il modello predice i logits per le classi, se le labels vengono fornite (come?) il modello si allena.
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=True,
        ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )



        #qui possiamo modificare l'output usato: per come è scritto usa solo l'ultimo layer hidden, possiamo concatenare gli ultimi 4
        #se concateni gli strati nascosti poi modifica il parametro hidden_size_lstm (1024*4=4096)
        #possiamo anche fare il pooling degli ultimi 4 strati per non moltiplicare troppo le dimensioni dell'embedding di input nella bilstm (che dovrà quindi essere raddoppiato a sua volta: 3072*2=6144)

        #ultimo livello di embedding
        sequence_output = outputs.last_hidden_state

        #ultimi 4 layers di embedding
        #last_four = outputs[2][-4:]
        #sequence_output = torch.cat((last_four[0], last_four[1], last_four[2], last_four[3]), 2)

        #normalizzazione dati per evitare overfitting
        sequence_output = self.dropout(sequence_output)

        # Bidirectional LSTM
        #lstm_out, _ = self.bilstm(sequence_output)

        #logits = self.classifier(lstm_out)
        logits=self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # aggiungi gli strati nascosti

        if labels is not None:
          loss_fct = nn.CrossEntropyLoss()
          # Mantieni solo la parte attiva della loss
          if attention_mask is not None:
              active_loss = attention_mask.view(-1) == 1
              active_logits = logits.view(-1, self.num_labels)
              active_labels = torch.where(
                  active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
              )
              loss = loss_fct(active_logits, active_labels)

          else:
              loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
          outputs = (loss,) + outputs
        return outputs
