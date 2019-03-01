import torch
import util
from torch import nn
import torch.nn.functional as F
from torch import optim
import time
import numpy as np
from .attention import EncoderRNN, Attn


class Base(nn.Module):
    def fit(self, loader_train, loader_valid, out, epochs=100, lr=1e-3):
        if 'optim' in self.__dict__:
            optimizer = self.optim
        else:
            optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = np.inf
        last_save = 0
        log = open(out + '.log', 'w')
        for epoch in range(epochs):
            t0 = time.time()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (1 - 1 / epochs) ** (epoch * 10)
            for i, (Xb, yb) in enumerate(loader_train):
                Xb, yb = Xb.to(util.dev), yb.to(util.dev)
                optimizer.zero_grad()
                y_ = self.forward(Xb, istrain=True)
                ix = yb == yb
                yb, y_ = yb[ix], y_[ix]
                loss = self.criterion(y_, yb)
                loss.backward()
                optimizer.step()
            loss_valid = self.evaluate(loader_valid)
            print('[Epoch: %d/%d] %.1fs loss_train: %f loss_valid: %f' % (
                    epoch, epochs, time.time() - t0, loss.item(), loss_valid), file=log)
            if loss_valid < best_loss:
                torch.save(self.state_dict(), out + '.pkg')
                print('[Performance] loss_valid is improved from %f to %f, Save model to %s' %
                      (best_loss, loss_valid, out + '.pkg'), file=log)
                best_loss = loss_valid
                last_save = epoch
            else:
                print('[Performance] loss_valid is not improved.', file=log)
                if epoch - last_save > 100: break
        log.close()
        self.load_state_dict(torch.load(out + '.pkg'))

    def evaluate(self, loader):
        loss = 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(util.dev), yb.to(util.dev)
            y_ = self.forward(Xb)
            ix = yb == yb
            yb, y_ = yb[ix], y_[ix]
            loss += self.criterion(y_, yb).item()
        return loss / len(loader)

    def predict(self, loader):
        score = []
        for Xb, yb in loader:
            Xb = Xb.to(util.dev)
            y_ = self.forward(Xb)
            score.append(y_.cpu().data)
        return torch.cat(score, dim=0).numpy()


class STClassifier(Base):
    def __init__(self, n_dim, n_class):
        super(STClassifier, self).__init__()
        self.dropout = nn.Dropout(0.25)
        self.fc0 = nn.Linear(n_dim, 4000)
        self.fc1 = nn.Linear(4000, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, n_class)
        if n_class == 1:
            self.criterion = nn.BCELoss()
            self.activation = nn.Sigmoid()
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.activation = nn.Softmax()
        self.to(util.dev)

    def forward(self, X, istrain=False):
        y = F.relu(self.fc0(X))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc1(y))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc2(y))
        if istrain:
            y = self.dropout(y)
        y = self.activation(self.fc3(y))
        return y


class MTClassifier(Base):
    def __init__(self, n_dim, n_task):
        super(MTClassifier, self).__init__()
        self.n_task = n_task
        self.dropout = nn.Dropout(0.25)
        self.fc0 = nn.Linear(n_dim, 8000)
        self.fc1 = nn.Linear(8000, 4000)
        self.fc2 = nn.Linear(4000, 2000)
        self.output = nn.Linear(2000, n_task)
        self.criterion = nn.BCELoss()
        self.activation = nn.Sigmoid()
        self.to(util.dev)

    def forward(self, X, istrain=False):
        y = F.relu(self.fc0(X))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc1(y))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc2(y))
        if istrain:
            y = self.dropout(y)
        y = self.activation(self.output(y))
        return y

    def inception(self, X):
        y = F.relu(self.fc0(X))
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        return y


class AttnClassifier(Base):
    def __init__(self, voc, emb_dim, h_dim, c_num):
        super(AttnClassifier, self).__init__()
        self.encoder = EncoderRNN(voc, emb_dim, h_dim)
        self.attn = Attn(h_dim * 2)
        self.out = nn.Linear(h_dim * 2, c_num)
        self.to(util.dev)
        self.criterion = nn.BCELoss()

    def forward(self, input, istrain=False):
        encoder_outputs = self.encoder(input)[0]
        attns = self.attn(encoder_outputs)  # (b, s, 1)
        feats = (encoder_outputs * attns).sum(dim=1)  # (b, s, h) -> (b, h)
        return F.sigmoid(self.out(feats))

