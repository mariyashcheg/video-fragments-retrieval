import torch
import torch.nn as nn

from data import EMBEDDING_DIM


class CALModel(nn.Module):

    def __init__(self, pretrained_emb, visual_input_dim, emb_dim=EMBEDDING_DIM, hidden_size=1000, dropout_rate=0.3):
        super(CALModel, self).__init__()
        self.hidden_size = hidden_size
        
        # visual 
        self.visual_fc = nn.Sequential(
            nn.Linear(in_features=visual_input_dim, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=emb_dim),
            nn.Dropout(p=dropout_rate)
        )            

        # language
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=pretrained_emb, freeze=True, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=pretrained_emb.size(1),
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.lang_fc = nn.Linear(in_features=self.hidden_size*2, out_features=emb_dim)
        
    def init_hidden(self, batch_size, device):
        return [torch.zeros(2, batch_size, self.hidden_size, device=device),
                torch.zeros(2, batch_size, self.hidden_size, device=device)]

    def forward(self, posit_features, intra_features, inter_features, lang_features, device):
        
        posit_emb = self.visual_fc(posit_features)
        intra_emb = self.visual_fc(intra_features)
        inter_emb = self.visual_fc(inter_features)
        
        embedded = self.word_embedding(lang_features)
        _, hidden = self.lstm(embedded, self.init_hidden(1, device))
        lang_emb = self.lang_fc(hidden[0].view(-1, 2*self.hidden_size))
        
        return posit_emb, intra_emb, inter_emb, lang_emb
