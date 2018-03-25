import torch.nn as nn
import torch
from torch.autograd import Variable

class Prior(nn.Module):
    def __init__(self, input_size, z_dim):
        super(Prior, self).__init__()
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden = nn.Linear(input_size, 512)
        self.mu = nn.Linear(512, z_dim)
        self.logvar = nn.Linear(512, z_dim)
        self.relu = nn.ReLU()

    def forward(self, input):
        def reparameterize(mu, logvar):
            logvar = logvar.mul(0.5).exp_()
            eps = Variable(logvar.data.new(logvar.size()).normal_())
            return eps.mul(logvar).add_(mu)

        hidden = self.relu(self.hidden(input))
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        z = reparameterize(mu, logvar)
        return z, mu, logvar


class Posterior(nn.Module):
    def __init__(self, input_size, z_dim):
        super(Posterior, self).__init__()
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden = nn.Linear(input_size, 512)
        self.mu = nn.Linear(512, z_dim)
        self.logvar = nn.Linear(512, z_dim)
        self.relu = nn.ReLU()

    def forward(self, input):
        def reparameterize(mu, logvar):
            logvar = logvar.mul(0.5).exp_()
            eps = Variable(logvar.data.new(logvar.size()).normal_())
            return eps.mul(logvar).add_(mu)
        input = torch.cat(input, dim=1)
        hidden = self.relu(self.hidden(input))
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        z = reparameterize(mu, logvar)
        return z, mu, logvar
