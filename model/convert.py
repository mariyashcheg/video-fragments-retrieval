import argparse
import torch
from torch import nn


class VaryModel(nn.Module):
    def __init__(self, vocab, emb_size, visual_input_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # visual
        self.visual_fc = nn.Sequential(
            nn.Linear(in_features=visual_input_dim, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=emb_size),
        )
        self.word_embedding = nn.Embedding(vocab, emb_size)
        # if self.normalize_lang:
        #     self.learnable_length = nn.Embedding.from_pretrained(
        #         embeddings=torch.ones(pretrained_emb.size(0), 1), freeze=False, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.lang_fc = nn.Linear(in_features=hidden_size * 2, out_features=emb_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        embedded = self.word_embedding(input)
        _, hidden = self.lstm(embedded)
        output = self.lang_fc(hidden[0].transpose(0, 1).reshape(input.size(0), 2 * self.hidden_size))
        return output


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="./best.pth")
    p.add_argument("--output", default="out.pth")
    p.add_argument("--part", default="textual")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ckpt = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    ddict = ckpt["model_state_dict"]
    # todo: take values from shapes
    emb_dim = 100
    vocab_size = 400002
    visual_input_dim = 8194
    hidden_size= 1000
    model = VaryModel(vocab_size, emb_dim, visual_input_dim, hidden_size)
    model.load_state_dict(ddict)
    scripted = torch.jit.script(model)
    scripted.save("./textual.zip")