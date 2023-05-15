import torch
import torch.nn as nn
from transformers import AutoModel


# when calculating loss, there should be space left for
# implementing the dependency loss, which punished the
# model for making predictions against the hierarchy
class RLModel(nn.Module):
    def __init__(self, model_config):
        # model_config contains:
        # 1. pretrained_model_name
        # 2. include_flat_prob
        # 3. n_labels (number of labels in the dataset, + 1 for [STOP], + 1 for root, + 1 for [PAD])
        # 4. label_embed_dim (we assign an embedding to each label)
        # 5. activation function (used in a Linear layer)

        super(RLModel, self).__init__()
        # Pretrained model. Can be BERT, RoBERTa, DeBERTa, etc.
        self.pretrained_model = AutoModel.from_pretrained(model_config.pretrained_model_name)
        self.n_labels = model_config.n_labels
        self.label_embed_dim = model_config.label_embed_dim
        self.act = model_config.act
        # extra labels to be included:
        self.label_embed = nn.Embedding(self.n_labels, self.label_embed_dim)
        # reproject the concatenated text embeddings & latest label embeddings to label_embed_dim
        self.reproject = nn.Linear(self.pretrained_model.config.hidden_size,
                                   self.label_embed_dim)
        self.cache = {'bid': None, 'embed': None}

    def forward(self, review_batch, next_labels_batch):
        # (batch_size, embed_dim)
        # for item in review_batch['encoding']:
        #     review_batch['encoding'][item] = review_batch['encoding'][item].view(1, -1)
        if self.cache['bid'] != review_batch['bid']:
            text_embed = self.pretrained_model(**review_batch['encoding']).last_hidden_state[:, 0, :].squeeze()
            self.cache['bid'] = review_batch['bid']
            self.cache['embed'] = text_embed
        else:
            text_embed = self.cache['embed']
        # text_embed = text_embed.view(1,-1)
        # (batch_size, embed_dim, 1)
        state_embed = self.act(self.reproject(text_embed)).unsqueeze(-1)

        # (batch_size, n_next_labels, hidden_dim)
        next_labels_embed = self.label_embed(next_labels_batch)

        # (batch_size, n_next_labels)
        return torch.bmm(next_labels_embed, state_embed).squeeze()

# A flat model, not in use anymore
class FlatModel(nn.Module):
    def __init__(self, model_config):
        super(FlatModel, self).__init__()
        # Pretrained model. Can be BERT, RoBERTa, DeBERTa, etc.
        self.pretrained_model = AutoModel.from_pretrained(model_config.pretrained_model_name)
        self.n_labels = model_config.n_labels
        self.reproject = nn.Linear(self.pretrained_model.config.hidden_size, self.n_labels)

    def forward(self, review_batch):
        return self.reproject(self.pretrained_model(**review_batch).last_hidden_state[:, 0, :].squeeze())
