import torch
import torch.nn as nn
from torch.autograd import Variable

class lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                #nn.BatchNorm1d(output_size),
                nn.Tanh())
        #self.hidden = self.init_hidden()

    def forward(self, input, t):
        def init_hidden():
            hidden = []
            for i in range(self.n_layers):
                hidden.append((Variable(torch.zeros(input.shape[0], self.hidden_size).cuda()),
                               Variable(torch.zeros(input.shape[0], self.hidden_size).cuda())))
            return hidden
        if t == 1:
            self.hidden = init_hidden()
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        return self.output(h_in)

class gaussian_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(gaussian_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        #self.hidden = self.init_hidden()

    def forward(self, input, t):
        def init_hidden():
            hidden = []
            for i in range(self.n_layers):
                hidden.append((Variable(torch.zeros(input.shape[0], self.hidden_size).cuda()),
                               Variable(torch.zeros(input.shape[0], self.hidden_size).cuda())))
            return hidden

        def reparameterize(mu, logvar):
            logvar = logvar.mul(0.5).exp_()
            eps = Variable(logvar.data.new(logvar.size()).normal_())
            return eps.mul(logvar).add_(mu)
        if t == 1:
            self.hidden = init_hidden()

        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = reparameterize(mu, logvar)
        return z, mu, logvar
            
