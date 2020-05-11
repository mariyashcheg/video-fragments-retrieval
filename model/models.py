import torch
import torch.nn as nn

from data import EMBEDDING_DIM


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.08, 0.08)
        torch.nn.init.constant_(m.bias, 0)

class CALModel(nn.Module):

    def __init__(self, visual_input_dim, pretrained_emb=None, emb_dim=EMBEDDING_DIM, hidden_size=1000, bert_emb=768,
                    dropout_rate=0.3, normalize_lang=False):
        super(CALModel, self).__init__()
        self.hidden_size = hidden_size
        self.normalize_lang = normalize_lang
        
        # visual 
        self.visual_fc = nn.Sequential(
            nn.Linear(in_features=visual_input_dim, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=emb_dim),
            nn.Dropout(p=dropout_rate)
        )
        self.visual_fc.apply(init_weights)            

        # language
        if pretrained_emb is None: # bert 
            self.lang_fc = nn.Linear(in_features=bert_emb, out_features=emb_dim)
        else: # glove
            self.word_embedding = nn.Embedding.from_pretrained(
                embeddings=pretrained_emb, freeze=True, padding_idx=0)

            if self.normalize_lang:
                self.learnable_length = nn.Embedding.from_pretrained(
                    embeddings=torch.ones(pretrained_emb.size(0), 1), freeze=False, padding_idx=0)
            
            self.lstm = nn.LSTM(
                input_size=pretrained_emb.size(1),
                hidden_size=self.hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            self.lang_fc = nn.Linear(in_features=self.hidden_size*2, out_features=emb_dim)
            self.lang_fc.apply(init_weights)
        
    def init_hidden(self, batch_size, device):
        return [torch.zeros(2, batch_size, self.hidden_size, device=device),
                torch.zeros(2, batch_size, self.hidden_size, device=device)]

    def forward(self, batch, visual=True, device=None, bert=False):
        if visual:
            output = self.visual_fc(batch)
        else:
            if bert:
                output = self.lang_fc(batch)
            else:
                embedded = self.word_embedding(batch)
                if self.normalize_lang:
                    length = self.learnable_length(batch)
                    embedded = embedded.div(embedded.norm(dim=-1, keepdim=True) + 1e-5)*length
                _, hidden = self.lstm(embedded, self.init_hidden(batch.size(0), device))
                output = self.lang_fc(hidden[0].transpose(0,1).reshape(batch.size(0), 2*self.hidden_size))

        return output