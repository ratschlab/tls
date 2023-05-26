import gin
import torch
import torch.nn as nn

from tls.models.layers import TransformerBlock, PositionalEncoding


@gin.configurable('GRU')
class GRUNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, num_classes, dropout=0.0, embedding_layer=gin.REQUIRED):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.embedding_layer = embedding_layer(input_dim, hidden_dim)

        self.rnn = nn.GRU(hidden_dim, hidden_dim, layer_dim, dropout=dropout, batch_first=True)
        self.logit = nn.Linear(hidden_dim, num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)
        return h0

    def forward(self, x):
        h0 = self.init_hidden(x)
        emb = self.embedding_layer(x)
        out, hn = self.rnn(emb, h0)
        pred = self.logit(out)
        return pred


@gin.configurable('Transformer')
class Transformer(nn.Module):
    def __init__(self, emb, hidden, heads, ff_hidden_mult, depth, num_classes, dropout=0.0,
                 pos_encoding=True, dropout_att=0.0, embedding_layer=gin.REQUIRED):
        super().__init__()
        self.embedding_layer = embedding_layer(emb, hidden)

        if pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden)
        else:
            self.pos_encoder = None

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(emb=hidden, hidden=hidden, heads=heads, mask=True,
                                            ff_hidden_mult=ff_hidden_mult,
                                            dropout=dropout, dropout_att=dropout_att))

        self.tblocks = nn.Sequential(*tblocks)
        self.logit = nn.Linear(hidden, num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.embedding_layer(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.tblocks(x)
        pred = self.logit(x)
        return pred
