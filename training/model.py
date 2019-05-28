import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.init as init


class ResidualBlock_1d(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock_1d, self).__init__()

        conv_block = [  nn.Linear(in_features, in_features),
                        nn.BatchNorm1d(in_features),
                        nn.ReLU(inplace=True),
                        nn.Linear(in_features, in_features),
                        nn.BatchNorm1d(in_features),
                        nn.ReLU(inplace=True)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
    
class ResidualBlock_1d_I(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock_1d_I, self).__init__()

        conv_block = [  nn.Linear(in_features, in_features),
                        nn.LayerNorm(in_features),
                        nn.ReLU(inplace=True),
                        nn.Linear(in_features, in_features),
                        nn.LayerNorm(in_features),
                        nn.ReLU(inplace=True)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator_1d(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=3):
        super(Generator_1d, self).__init__()

        # Initial convolution block       
        model = [   nn.Linear(input_nc, 64),
                    nn.LayerNorm(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(1):
            model += [  nn.Linear(in_features, out_features),
                        nn.LayerNorm(out_features),
                        nn.ReLU(inplace=True) ] # T 512 18 13
            in_features = out_features
            out_features = in_features*2 
        in_features += 100 # 228
        out_features = 4096 # 512
        # Residual blocks
        model2 = [ nn.Linear(in_features, out_features),
                   nn.LayerNorm(out_features),
                   nn.ReLU(inplace=True)]
        in_features = out_features
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock_1d_I(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(1):
            model2 += [ nn.Linear(in_features, out_features),
                        nn.LayerNorm(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model2 += [ nn.Dropout(0.5),
                    nn.Linear(in_features, output_nc) ]

        self.model = nn.Sequential(*model)
        self.model2 = nn.Sequential(*model2)

    def forward(self, x, noise):
        x = self.model(x)
        x = torch.cat([x, noise], 1)
        return self.model2(x)
    
class Generator_V(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, T, dropout=0, gpu=True):
        super(Generator_V, self).__init__()

        self.output_size      = output_size
        self._gpu        = gpu
        self.hidden_size = hidden_size
        self.T = T

        # define layers
        self.gru    = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, bidirectional=True) #dropout=0.5, 
        self.linear = nn.Linear(hidden_size*2, self.output_size)
        self.bn     = nn.BatchNorm1d(self.output_size, affine=False)

    def forward(self, input):
        '''
        inputs.shape()   => (batch_size, input_size)
        outputs.shape() => (seq_len, batch_size, output_size)
        '''
        outputs, (self.hidden, self.hidden_c) = self.gru(input, (self.hidden, self.hidden_c))
        outputs_reshaped = outputs.contiguous().view(-1, self.hidden_size*2)
        outputs_reshaped = self.linear(outputs_reshaped)
        return outputs_reshaped

    def initWeight(self, init_forget_bias=1):
        # See details in https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
        for name, params in self.named_parameters():
            if 'weight' in name:
                init.xavier_uniform(params)

            # initialize forget gate bias
            elif 'gru.bias_ih_l' in name:
                b_ir, b_iz, b_in = params.chunk(3, 0)
                init.constant(b_iz, init_forget_bias)
            elif 'gru.bias_hh_l' in name:
                b_hr, b_hz, b_hn = params.chunk(3, 0)
                init.constant(b_hz, init_forget_bias)
            else:
                init.constant(params, 0)

    def initHidden(self, batch_size):
        self.hidden = Variable(torch.zeros(6, batch_size, self.hidden_size)).cuda()
        self.hidden_c = Variable(torch.zeros(6, batch_size, self.hidden_size)).cuda()

class Discriminator_1d(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=3):
        super(Discriminator_1d, self).__init__()

        # Initial convolution block       
        model = [   nn.Linear(input_nc, 4096),
                    nn.BatchNorm1d(4096),
                    nn.ReLU(inplace=True) ]

        in_features = 4096 # 228
        out_features = 4096
        
        in_features = out_features
        for _ in range(n_residual_blocks):
            model += [ResidualBlock_1d(in_features)]

        # Output layer
        model += [ nn.Dropout(0.5),
                   nn.Linear(in_features, output_nc) ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
class Discriminator_2d(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, T, dropout=0, gpu=True):
        super(Discriminator_2d, self).__init__()

        self.output_size      = output_size
        self._gpu        = gpu
        self.hidden_size = hidden_size
        self.T = T

        # define layers
        self.gru    = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
        linear = [  nn.Linear(hidden_size*2*T, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, self.output_size),
                ]
        self.linear = nn.Sequential(*linear)
        self.bn     = nn.BatchNorm1d(self.output_size, affine=False)

    def forward(self, input):
        '''
        inputs.shape()   => (batch_size, input_size)
        outputs.shape() => (seq_len, batch_size, output_size)
        '''
        outputs, (self.hidden, self.hidden_c) = self.gru(input, (self.hidden, self.hidden_c))
        outputs_reshaped = outputs.contiguous().view(-1, self.hidden_size*2*self.T)
        outputs_reshaped = self.linear(outputs_reshaped)
        return outputs_reshaped

    def initWeight(self, init_forget_bias=1):
        # See details in https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
        for name, params in self.named_parameters():
            if 'weight' in name:
                init.xavier_uniform(params)

            # initialize forget gate bias
            elif 'gru.bias_ih_l' in name:
                b_ir, b_iz, b_in = params.chunk(3, 0)
                init.constant(b_iz, init_forget_bias)
            elif 'gru.bias_hh_l' in name:
                b_hr, b_hz, b_hn = params.chunk(3, 0)
                init.constant(b_hz, init_forget_bias)
            else:
                init.constant(params, 0)

    def initHidden(self, batch_size):
        self.hidden = Variable(torch.zeros(4, batch_size, self.hidden_size)).cuda()
        self.hidden_c = Variable(torch.zeros(4, batch_size, self.hidden_size)).cuda()
    
class Encoder_1d(nn.Module):
    def __init__(self, nc, nc_noise, conv_dim=64):
        super(Encoder_1d, self).__init__() # b nc T img_size, img_size
        self._name = 'discriminator_wgan'

        layers = []
        layers.append(nn.Linear(nc, conv_dim))
        layers.append(nn.BatchNorm1d(conv_dim))
        layers.append(nn.ReLU())
        
        curr_dim = conv_dim
        layers.append(nn.Linear(curr_dim, curr_dim*2))
        layers.append(nn.BatchNorm1d(curr_dim*2))
        layers.append(nn.ReLU())
        
        curr_dim = curr_dim * 2
        layers.append(nn.Linear(curr_dim, curr_dim*2))
        layers.append(nn.BatchNorm1d(curr_dim*2))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(curr_dim*2, curr_dim*2))
        layers.append(nn.BatchNorm1d(curr_dim*2))
        layers.append(nn.ReLU())
        
        curr_dim = curr_dim * 2
        layers.append(nn.Linear(curr_dim, curr_dim*2))
        layers.append(nn.BatchNorm1d(curr_dim*2))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(curr_dim*2, curr_dim*2))
        layers.append(nn.BatchNorm1d(curr_dim*2))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(curr_dim*2, curr_dim*2))
        layers.append(nn.BatchNorm1d(curr_dim*2))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(curr_dim*2, curr_dim*2))
        layers.append(nn.BatchNorm1d(curr_dim*2))
        layers.append(nn.ReLU())
        
        self.mu = nn.Linear(curr_dim*2, nc_noise)
        self.std = nn.Linear(curr_dim*2, nc_noise)

        self.main = nn.Sequential(*layers)
        #self.tanh = nn.Tanh()
    def encode(self, x):
        h = self.main(x)
        return self.mu(h), self.std(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x) # b, 100, 18, 13
        z = self.reparameterize(mu, logvar)
        #out_real = self.tanh(h)
        return z, mu, logvar
