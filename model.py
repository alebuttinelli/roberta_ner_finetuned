# -*- coding: utf-8 -*-
"""
Defines the custom RoBERTaforNER model architecture.
This file contains the PyTorch class for the model.
"""

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput

class RoBERTaforNER(BertPreTrainedModel):
    """
    Custom XLM-RoBERTa model for Token Classification (NER) with an optional
    BiLSTM layer.
    """

    def __init__(self, config, num_labels, use_bilstm=True, lstm_hidden_size=1024, num_lstm_layers=2):
        super().__init__(config)
        self.num_labels = num_labels
        self.use_bilstm = use_bilstm

        # Load the base model
        self.roberta = AutoModel.from_pretrained(config._name_or_path, config=config, add_pooling_layer=False)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Optional BiLSTM Layer
        if self.use_bilstm:
            self.bilstm = nn.LSTM(
                input_size=config.hidden_size,
                hidden_size=lstm_hidden_size,
                num_layers=num_lstm_layers,
                bidirectional=True,
                batch_first=True
            )
            classifier_input_size = lstm_hidden_size * 2
        else:
            self.bilstm = None
            classifier_input_size = config.hidden_size

        # Classifier Head
        self.classifier = nn.Linear(classifier_input_size, self.num_labels)

        # Initialize weights (inherited function)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        # Base Model Pass
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # BiLSTM Pass (Optional)
        if self.use_bilstm and self.bilstm is not None:
            lstm_out, _ = self.bilstm(sequence_output)
            logits = self.classifier(lstm_out)
        else:
            logits = self.classifier(sequence_output)

        # Loss Calculation
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Return Standard Output Object
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
