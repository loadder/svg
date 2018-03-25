import torch.nn as nn
import torch
from torch.autograd import Variable

class generation(nn.Module):
    def __init__(self, encoder, posterior, prior, decoder, skip_connection=True):
        super(generation, self).__init__()
        self.encoder = encoder
        self.posterior = posterior
        self.prior = prior
        self.decoder = decoder
        self.rnn = nn.LSTMCell(encoder.dim, prior.input_size)
        self.skip = skip_connection
        self.z_dim = prior.mu.out_features

    def forward(self, input, t):
        if t == 0:
            self.hidden = (Variable(torch.zeros(input.shape[0], self.rnn.hidden_size).cuda()),
                               Variable(torch.zeros(input.shape[0], self.rnn.hidden_size).cuda()))
        encoded_x, skip = self.encoder(input)
        self.hidden = self.rnn(encoded_x, self.hidden)
        self.h = self.hidden[0]
        posterior_z, posterior_mu, posterior_logvar = self.posterior([self.h, encoded_x])
        if t == 0:
            prior_mu = Variable(torch.zeros([input.shape[0], self.z_dim])).cuda()
            prior_logvar = Variable(torch.zeros([input.shape[0], self.z_dim])).cuda()
            prior_z = Variable(torch.zeros([input.shape[0], self.z_dim]).normal_()).cuda()
        else:
            prior_z, prior_mu, prior_logvar = self.prior(self.h)
        z = posterior_z if self.training else prior_z
        pred_frame = self.decoder([z, skip])
        return pred_frame, prior_mu, prior_logvar, posterior_mu, posterior_logvar
