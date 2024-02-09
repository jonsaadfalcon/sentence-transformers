

import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from ..SentenceTransformer import SentenceTransformer
import random


class OrthogonalProjectionLoss(nn.Module):
    
    def __init__(self, model: SentenceTransformer, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(OrthogonalProjectionLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        if self.model.max_seq_length in [512]: # for BGE and other baselines
            embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
            output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
            return self.loss_fct(output, labels.view(-1))
        else:
            embeddings = [self.model[0]._modules['auto_model'](input_ids=sentence_feature['input_ids'], attention_mask=sentence_feature['attention_mask'], token_type_ids=sentence_feature['token_type_ids'])['sentence_embedding'] for sentence_feature in sentence_features]
            output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
            loss = self.loss_fct(output.to(torch.bfloat16), labels.view(-1).to(torch.bfloat16))
            return loss