import math
import torch
import util
from torch import nn
import torch.nn.functional as F
from torch import optim
from .attention import EncoderRNN, DecoderRNN, Attention


class Base(nn.Module):
    def fit(self, pair_loader, tgt_loader, epochs=100, out=None):
        log = open(out + '.log', 'w')
        best_valid = 0.
        net = nn.DataParallel(self, device_ids=util.devices)
        optimizer = torch.optim.Adam(self.parameters())
        for epoch in range(epochs):
            for i, (tgts, cmps) in enumerate(pair_loader):
                tgts, cmps = tgts.to(util.dev), cmps.to(util.dev)
                optimizer.zero_grad()
                output = net(tgts, cmps)
                loss = F.nll_loss(output.view(-1, self.voc_cmp.size), cmps.view(-1))
                loss.backward()
                optimizer.step()

                if i % 10 != 0 or i == 0: continue
                ids, smiles, valids = [], [], []
                for _ in range(4):
                    for ix, tgt in tgt_loader:
                        seqs = net(tgt.to(util.dev))
                        # ix = util.unique(seqs)
                        # seqs = seqs[ix]
                        smile, valid = util.check_smiles(seqs, self.voc_cmp)
                        smiles += smile
                        valids += valid
                        ids += ix.tolist()
                valid = sum(valids) / len(valids)
                print("Epoch: %d step: %d loss: %.3f valid: %.3f" % (epoch, i, loss.item(), valid), file=log)
                for i, smile in enumerate(smiles):
                    print('%d\t%s' % (valids[i], smile), file=log)
                if best_valid < valid:
                    torch.save(self.state_dict(), out + '.pkg')
                    best_valid = valid
        log.close()


class DecoderAttn(nn.Module):
    def __init__(self, voc, embed_size, hidden_size, n_layers=3):
        super(DecoderAttn, self).__init__()
        self.voc = voc
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = voc.size
        self.n_layers = n_layers

        self.embed = nn.Embedding(voc.size, embed_size)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(embed_size  + hidden_size, hidden_size, num_layers=self.n_layers)
        self.out = nn.Linear(hidden_size, voc.size)

    def forward(self, input, h, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input.unsqueeze(0)) # (1,B,N)
        # Calculate attention weights and apply to encoder outputs
        attn_w = self.attention(h[0], encoder_outputs)
        context = attn_w.bmm(encoder_outputs)  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        output = torch.cat([embedded, context], 2)
        output, h_out = self.gru(output, h)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        # context = context.squeeze(0)
        # output = self.out(torch.cat([output, context], dim=1))
        output = self.out(output)
        return output, h_out


class Seq2Seq(Base):
    def __init__(self, voc_tgt, voc_cmp):
        super(Seq2Seq, self).__init__()
        self.hidden_size = 256
        self.voc_tgt = voc_tgt
        self.voc_cmp = voc_cmp
        self.encoder = EncoderRNN(voc_tgt, 5, self.hidden_size)
        self.decoder = DecoderAttn(voc_cmp, 128, self.hidden_size * 2)

        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.NLLLoss()

    def forward(self, tgts, cmps=None, cutoff=0.9):
        batch_size, seq_len = tgts.size()
        hidden = torch.zeros(3, batch_size, self.hidden_size * 2).to(util.dev)
        encoder_outputs, hidden[0] = self.encoder(tgts, None)

        output = torch.zeros(batch_size, self.voc_cmp.max_len).long().to(util.dev)
        if cmps is not None:
            output = torch.zeros(batch_size, self.voc_cmp.max_len, self.voc_cmp.size)
        # Start token
        x = torch.LongTensor([self.voc_cmp.vocab['GO']] * batch_size).to(util.dev)
        isEnd = torch.zeros(batch_size).byte().to(util.dev)
        for step in range(self.voc_cmp.max_len):
            logit, hidden = self.decoder(x, hidden, encoder_outputs)
            if cmps is not None:
                score = F.log_softmax(logit, dim=-1)
                output[:, step, :] = score
                x = cmps[:, step]
            else:
                proba = F.softmax(logit, dim=-1)
                x = torch.multinomial(proba, 1).view(-1)
                x[isEnd] = self.voc_cmp.vocab['EOS']
                output[:, step] = x
        return output


class EncDec(Base):
    def __init__(self, voc_tgt, voc_cmp):
        super(EncDec, self).__init__()
        self.voc_size = 128
        self.hidden_size = 256
        self.voc_tgt = voc_tgt
        self.voc_cmp = voc_cmp
        self.encoder = EncoderRNN(voc_tgt, 5, self.hidden_size)
        self.decoder = DecoderRNN(voc_cmp, self.voc_size, self.hidden_size * 2)

    def forward(self, tgts, cmps=None):
        batch_size = tgts.size(0)
        hidden = self.decoder.init_h(batch_size)
        hidden[0] = self.encoder(tgts)[1]
        output = torch.zeros(batch_size, self.voc_cmp.max_len).long().to(util.dev)
        if cmps is not None:
            output = torch.zeros(batch_size, self.voc_cmp.max_len, self.voc_cmp.size)

        x = torch.LongTensor([self.voc_cmp.vocab['GO']] * batch_size).to(util.dev)
        isEnd = torch.zeros(batch_size).byte().to(util.dev)
        for step in range(self.voc_cmp.max_len):
            logit, hidden = self.decoder(x, hidden)
            if cmps is not None:
                score = F.log_softmax(logit, dim=-1)
                output[:, step, :] = score
                x = cmps[:, step]
            else:
                proba = F.softmax(logit, dim=-1)
                x = torch.multinomial(proba, 1).view(-1)
                x[isEnd] = self.voc_cmp.vocab['EOS']
                output[:, step] = x
        return output