import torch
import torch.nn as nn

from data import EMBEDDING_DIM, FRAMES_PER_SEC, SEC_PER_SEGMENT
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FrameAggration(nn.Module):

    def __init__(self, aggr_type, input_dim, hidden_dim, out_dim, pooling='avg'):
        super(FrameAggration, self).__init__()
        self.aggr_type = aggr_type

        if pooling == 'avg':
            AdaptivePool = nn.AdaptiveAvgPool2d
            Pool = nn.AvgPool2d
        else:
            AdaptivePool = nn.AdaptiveMaxPool2d
            Pool = nn.MaxPool2d

        if self.aggr_type == 'conv_pooling':
            self.aggr_block = nn.Sequential(
                AdaptivePool(output_size=(1,input_dim)),
                nn.Linear(in_features=input_dim, out_features=hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=out_dim)
            )
        elif self.aggr_type == 'late_pooling':
            self.aggr_block = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=out_dim),
                AdaptivePool(output_size=(1,out_dim))
            )
        elif self.aggr_type == 'slow_pooling':
            self.aggr_block = nn.Sequential(
                Pool(kernel_size=(10, 1), stride=(5,1)),
                nn.Linear(in_features=input_dim, out_features=hidden_dim),
                nn.ReLU(),
                AdaptivePool(output_size=(1,hidden_dim)),
                nn.Linear(in_features=hidden_dim, out_features=out_dim)
            )
    
    def forward(self, x):
        x = self.aggr_block(x).squeeze(-2)
        return x


class CALModel(nn.Module):

    def __init__(self, visual_input_dim, lang_input_dim=100, embedding_dim=EMBEDDING_DIM,
            visual_hidden_dim=500, lang_hidden_dim=1000, aggr_type='conv_pooling', pooling='avg',
            bidir=True, bert=False, hidden=False, max_query_len=20, dropout_rate=0.3):

        super(CALModel, self).__init__()
        self.lang_hidden_dim = lang_hidden_dim
        self.max_query_len = max_query_len
        self.bert = bert
        self.hidden = hidden
        
        # visual 
        self.visual_encoder = FrameAggration(
            aggr_type=aggr_type, 
            input_dim=visual_input_dim,
            hidden_dim=visual_hidden_dim, 
            out_dim=embedding_dim, 
            pooling=pooling
        )
        self.visual_fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=embedding_dim*2+2, out_features=embedding_dim),
            # nn.Dropout(p=dropout_rate)
        )

        # language
        if self.bert: 
            self.lang_fc = nn.Linear(in_features=lang_input_dim, out_features=embedding_dim)
        else: # glove
            self.lstm = nn.LSTM(
                input_size=lang_input_dim,
                hidden_size=self.lang_hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=bidir
            )
            self.lang_fc = nn.Linear(in_features=self.lang_hidden_dim*(int(bidir)+1), out_features=embedding_dim)
        
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.data.fill_(0)
            elif 'weight' in name:
                param.data.uniform_(-0.08, 0.08)

    def forward(self, batch, visual=True, endpoints=None, len_seq=None, device=None):
        if visual:
            context, moments_batch = [], []
            for b in range(len(batch)):
                moments = []
                for i in range(endpoints[b, 0], endpoints[b, 1]+1):
                    # [1, nframes, visual_emb_dim]
                    moments.append(
                        batch[b][i*FRAMES_PER_SEC*SEC_PER_SEGMENT:(i+1)*FRAMES_PER_SEC*SEC_PER_SEGMENT, :].unsqueeze(0)
                        ) 
                # list of [1, len_moment, nframes, visual_emb_dim]
                moments_batch.append(torch.cat(moments, axis=0).unsqueeze(0))
                # list of [1, 1, nframes, visual_emb_dim]
                context.append(
                    self.visual_encoder(
                        batch[b][:min(batch[b].size(0), FRAMES_PER_SEC*SEC_PER_SEGMENT*6), :].unsqueeze(0).unsqueeze(0).to(device))
                    )
            
            # [bs, 1, 1, visual_emb_dim] -> [bs, 1, emb_dim]
            context = torch.cat(context, axis=0)
            # [bs, len_moment, nframes, visual_emb_dim] -> [bs, len_moment, emb_dim]
            segments = self.visual_encoder(torch.cat(moments_batch, axis=0).to(device))
            temporal_endpoints = endpoints.unsqueeze(1).float() / 6.
            features = torch.cat(
                [
                    segments, 
                    context.repeat(1, segments.size(1), 1),
                    temporal_endpoints.repeat(1, segments.size(1), 1)
                ], axis=-1)
            output = self.visual_fc(features)
            
        else:
            if self.bert:
                output = self.lang_fc(batch)
            else:
                packed_embedded = pack_padded_sequence(batch, len_seq, batch_first=True, enforce_sorted=False)
                packed_output, hidden = self.lstm(packed_embedded)
                if self.hidden:
                    output = self.lang_fc(hidden[0].transpose(0,1).reshape(batch.size(0), 2*self.lang_hidden_dim))
                else:
                    lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=self.max_query_len)
                    output = self.lang_fc(lstm_output[range(batch.size(0)), len_seq - 1, :])

        return output