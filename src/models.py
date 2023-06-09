import torch
import torch.nn as nn
import itertools
from pyod.models import base as pyodBase

class AttentionNet(nn.Module):
    def _init_weights(self, module):
        if self.initialization == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        elif self.initialization == 'xavier_normal':
            torch.nn.init.xavier_normal_(module.weight, gain=torch.nn.init.calculate_gain(nonlinearity='linear'))
        elif self.initialization == 'ones':
            torch.nn.init.ones_(module.weight)
        elif self.initialization == 'zeros':
            torch.nn.init.zeros_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
        
    def initAllWeights(self):
        for layer in self.children():
            if type(layer) != nn.Sequential:
                self._init_weights(layer)
        self.weights = (torch.diag(torch.ones(self.E_Dim)))
        
    def __init__(self, num_feature, sample_size, E_dim:int=8, initialization='xavier_normal', **kwargs) -> None:
        super().__init__()
        self.name = 'AttentionNet'
        self.initialization = initialization
        self.E_Dim, self.sample_size = E_dim, sample_size
        #self.weights = 0
        self.getE = nn.Sequential(
            nn.Linear(num_feature, self.E_Dim, bias=True),
            nn.Tanh(),
        )
        self.att1 = nn.Linear(self.sample_size, 128)
        self.att2 = nn.Linear(128, 16)
        self.att3 = nn.Linear(16, 1)
        self.weights = (torch.diag(torch.ones(self.E_Dim)))
        
        self.initAllWeights()
        
        self.attNet = nn.Sequential(
            self.att1,
            nn.LeakyReLU(0.1),
            self.att2,
            nn.LeakyReLU(0.1),
            self.att3,
            nn.Tanh(),
        )
        
    def forward(self, x):
        E = (self.getE(x)) # N x E dimension
        #E = x
        weights = torch.softmax(torch.flatten(self.attNet(torch.t(E))), dim=0) * self.E_Dim # E x 1 weights
        weights = torch.diag(weights) # transform to a E x E diagonal matrix
        self.weights = (weights.detach().clone())
        x = torch.matmul(E, weights)
        return x
    
    def transformData(self, x):
        #return torch.matmul(x, self.weights)
        return torch.matmul((self.getE(x)), self.weights)

class AutoEncoder(nn.Module):
    def _init_weights(self, module):
        if isinstance(module, nn.Sequential):
            for layer in module:
                self._init_weights(layer)
        if not isinstance(module, nn.Linear):
            return
        if self.initialization == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        elif self.initialization == 'xavier_normal':
            torch.nn.init.xavier_normal_(module.weight, gain=torch.nn.init.calculate_gain(nonlinearity='linear'))
        elif self.initialization == 'ones':
            torch.nn.init.ones_(module.weight)
        elif self.initialization == 'zeros':
            torch.nn.init.zeros_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()

    def __init__(self, num_feature, output_feature=0, latent_dim=4, hidden_dim=8, activation='leaky_relu', initialization='xavier_normal', logit=False, layer_config=None, **kwargs) -> None:
        super().__init__()
        self.name = 'AE' # name
        self.num_feature, self.latent_dim, self.hidden_dim, self.initialization, self.logit = num_feature, latent_dim, hidden_dim, initialization, logit # arguments
        self.output_feature = num_feature if output_feature == 0 else output_feature
        if logit:
            self.output_feature = 1
        self.hd = hd = hidden_dim
        """
        self.queryDim, self.valueDim = 10, num_feature
        self.getQ = nn.Linear(num_feature, self.queryDim, bias=True)
        self.getK = nn.Linear(num_feature, self.queryDim, bias=True)
        self.getV = nn.Linear(num_feature, self.valueDim, bias=True) """
        self.error = None # extra error terms, if applicable
        self.NS = False
        self.naiveScaler= nn.Parameter(torch.diag(torch.rand(self.num_feature))) if self.NS else torch.diag(torch.ones(self.num_feature))
        """ self.transformerAttention = False

        if self.transformerAttention:
            encoding_input_dim = self.valueDim
        else:
            encoding_input_dim = num_feature """
        encoding_input_dim = num_feature
        # check if layer config is specified
        if not layer_config:
            self.layer_config = [[64, 64, 16, 16, 16, 16, latent_dim, latent_dim], [latent_dim, latent_dim, 16, 16, 16, 16, 64, 64]]
        else:
            self.layer_config = layer_config
        e_config, d_config = [encoding_input_dim] + self.layer_config[0], self.layer_config[1] + [self.output_feature]
        e_layers = [nn.Linear(e_config[i], e_config[i+1]) for i, _ in enumerate(e_config[:-1])]
        d_layers = [nn.Linear(d_config[i], d_config[i+1]) for i, _ in enumerate(d_config[:-1])]
        e_activation, d_activation = [nn.LeakyReLU(0.1) for _ in range(len(e_layers) - 1)], [nn.LeakyReLU(0.1) for _ in range(len(d_layers) - 1)]
        if activation == 'leaky_relu':
            e_activation, d_activation = [nn.LeakyReLU(0.1) for _ in range(len(e_layers) - 1)], [nn.LeakyReLU(0.1) for _ in range(len(d_layers) - 1)]
            # initialize encoder and decoder
            self.encoding_layer = nn.Sequential(*list(itertools.chain(*itertools.zip_longest(e_layers, e_activation)))[:-1])
            self.decoding_layer = nn.Sequential(*list(itertools.chain(*itertools.zip_longest(d_layers, d_activation)))[:-1])
            
            """ self.l1 = nn.Linear(encoding_input_dim, self.hidden_dim)
            self.l2 = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.l3 = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.l4 = nn.Linear(self.hidden_dim, self.latent_dim)
            self.l5 = nn.Linear(self.latent_dim, self.hidden_dim)
            self.l6 = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.l7 = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.l8 = nn.Linear(self.hidden_dim, self.output_feature)
            self.encoding_layer = nn.Sequential(
                self.l1,
                nn.LeakyReLU(0.1),
                self.l2,
                nn.LeakyReLU(0.1),
                self.l3,
                nn.LeakyReLU(0.1),
                self.l4,
            )  
            self.decoding_layer = nn.Sequential(
                self.l5,
                nn.LeakyReLU(0.1),
                self.l6,
                nn.LeakyReLU(0.1),
                self.l7,
                nn.LeakyReLU(0.1),
                self.l8
            ) """
            
        elif activation == 'tanh':
            self.encoding_layer = nn.Sequential(
                self.l1,
                nn.Tanh(),
                self.l2,
                nn.Tanh(),
                self.l3,
                nn.Tanh(),
                self.l4,
            )  
            self.decoding_layer = nn.Sequential(
                self.l5,
                nn.Tanh(),
                self.l6,
                nn.Tanh(),
                self.l7,
                nn.Tanh(),
                self.l8
            )
        elif activation == 'sigmoid':
            self.encoding_layer = nn.Sequential(
                self.l1,
                nn.Sigmoid(),
                self.l2,
                nn.Sigmoid(),
                self.l3,
                nn.Sigmoid(),
                self.l4,
            )  
            self.decoding_layer = nn.Sequential(
                self.l5,
                nn.Sigmoid(),
                self.l6,
                nn.Sigmoid(),
                self.l7,
                nn.Sigmoid(),
                self.l8,
            )
        else: # for now, use leaky relu as default
            self.encoding_layer = nn.Sequential(
                self.l1,
                nn.LeakyReLU(0.1),
                self.l2,
                nn.LeakyReLU(0.1),
                self.l3,
                nn.LeakyReLU(0.1),
                self.l4,
            )  
            self.decoding_layer = nn.Sequential(
                self.l5,
                nn.LeakyReLU(0.1),
                self.l6,
                nn.LeakyReLU(0.1),
                self.l7,
                nn.LeakyReLU(0.1),
                self.l8,
            )
        # initialize all weights
        for layer in self.children():
            self._init_weights(layer)
        
    def transformer(self, x):
        Q, K, V = self.getQ(x), self.getK(x), self.getV(x)
        attention_score = torch.softmax(torch.matmul(Q, torch.t(K)), dim=-1)
        x = torch.tanh(torch.matmul(attention_score, V))
        return x
    
    def encode(self, x):
        if self.NS:
            x = torch.matmul(x, self.naiveScaler)
        """ if self.transformerAttention:
            x = self.transformer(x) """
        return self.encoding_layer(x)
    
    def decode(self, x):
        x = self.decoding_layer(x) if not self.logit else nn.Sigmoid()(self.decoding_layer(x))
        if self.NS:
            x = torch.matmul(x, torch.linalg.inv(self.naiveScaler))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))

class Discriminator(AutoEncoder):
    def __init__(self, num_feature, latent_dim=64, hidden_dim=16, activation='leaky_relu', initialization='xavier_normal', **kwargs) -> None:
        super().__init__(num_feature=num_feature, output_feature=1, hidden_dim=hidden_dim, latent_dim=latent_dim, activation=activation, initialization=initialization, **kwargs)
        self.name = 'Discriminator'
        self.sig = nn.Sigmoid()
    
    def encode(self, x):
        return self.encoding_layer(x)
    def decode(self, x):
        return self.sig(self.decoding_layer(x))
    def forward(self, x):
        return self.decode(self.encode(x))
    def l_output(self, x):
        return (self.decoding_layer(self.encoding_layer(x)))

class VAE(AutoEncoder):
    def __init__(self, num_feature, output_feature=0, latent_dim=4, hidden_dim=8, activation='leaky_relu', initialization='xavier_normal', sigma=1, layer_config=None, **kwargs) -> None:
        super().__init__(num_feature=num_feature, output_feature=output_feature, latent_dim=latent_dim, hidden_dim=hidden_dim, activation=activation, initialization=initialization, layer_config=layer_config)
        self.name = 'VAE'
        self.num_feature, self.latent_dim, self.sigma, self.initialization = num_feature, latent_dim, sigma, initialization
        if self.layer_config:
            self.latent_dim = self.layer_config[0][-1]
        self.hd = hd = hidden_dim
        self.KLD = 0
        self.standard_multivariateNormal = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(latent_dim), torch.eye(latent_dim))
        self.generate_mean = nn.Linear(self.latent_dim, self.latent_dim)
        self.generate_log_var = nn.Linear(self.latent_dim, self.latent_dim)
        self.generate_std = nn.Linear(self.latent_dim, self.latent_dim)
        self._init_weights(self.generate_mean)
        self._init_weights(self.generate_std)
        self._init_weights(self.generate_log_var)
    
    def reparameterize(self, mu, var):
        std = torch.sqrt(var) # standard deviation
        #var = std**2
        # re-parameterization trick, as we introduce a new random eps instead of back-propagate through a random-sampling
        eps = torch.randn_like(std)
        sample = mu + (eps * std) 
        self.KLD = -0.5 * torch.mean(1 + torch.log(var) - mu**2 - var)
        # KL divergence with N(0, I_d), assuming all latent variables are independent
        #self.KLD = 0.5 * self.sigma * torch.sum(torch.log(torch.prod(var, dim=1, keepdim=True)) - self.latent_dim + torch.sum(1/var, dim=1, keepdim=True) + torch.sum(mu**2/var, dim=1, keepdim=True))
        self.error = self.sigma * self.KLD.sum()
        return sample

    # This seems very tricky to implement under current framework of pytorch.    
    """ def reparameterize_multivariate(self, Mu, Sigma):
        p = torch.distributions.multivariate_normal.MultivariateNormal(Mu, Sigma)
        eps = torch.randn_like(Mu) # `randn_like` as we need the same size: B x L
        L = torch.linalg.cholesky(Sigma)# compute Cholesky decomposition for reparameterization
        sample = torch.mul(L, eps) + Mu
        self.KLD = torch.distributions.kl.kl_divergence(p, self.standard_multivariateNormal) # KL divergence with N(0, I)
        self.error = self.sigma * self.KLD.sum()
        return sample """
        
    def encode(self, x):
        """ if self.transformerAttention:
            Q, K, V = self.getQ(x), self.getK(x), self.getV(x)
            attention_score = torch.softmax(torch.matmul(Q, torch.t(K)), dim=-1)
            x = torch.tanh(torch.matmul(attention_score, V)) """
        x = self.encoding_layer(x)
        # simplified distribution generation
        mu, var = self.generate_mean(x), torch.exp(10*nn.Tanh()(self.generate_log_var((x))))
        return self.reparameterize(mu, var)

class DAE(AutoEncoder):
    def __init__(self, num_feature, latent_dim=4, hidden_dim=8, activation='leaky_relu', initialization='xavier_normal', noise_factor=0.1, noise_fraction=0.5, **kwargs) -> None:
        super().__init__(num_feature=num_feature, output_feature=0, hidden_dim=hidden_dim, latent_dim=latent_dim, activation=activation, initialization=initialization)
        self.name = 'DAE'
        self.noise_factor = noise_factor
        self.noise_fraction = noise_fraction

class DOCAE(AutoEncoder):
    def __init__(self, num_feature, output_feature=0, center:torch.Tensor=torch.tensor([[0]]), v:float=1e-1, R:torch.Tensor=torch.tensor(1, dtype=float, requires_grad=True), latent_dim=4, hidden_dim=8, activation='leaky_relu', initialization='xavier_normal', alpha=1, **kwargs) -> None:
        super().__init__(num_feature=num_feature, output_feature=output_feature, hidden_dim=hidden_dim, latent_dim=latent_dim, activation=activation, initialization=initialization)
        if not 0 < v <= 1:
            raise ValueError('Tradeoff v should be in range (0, 1]')
        self.name = 'DOCAE'
        self.center = center
        self.v = v
        self.R = nn.Parameter(R)
        self.alpha = alpha
        #self.R = R
        self.error = 0
        self.queryDim, self.valueDim = 16, 8

    def encode(self, x):
        code = self.encoding_layer(x)
        distance = torch.sum((code - self.center.expand(code.shape[0], code.shape[1])) ** 2, dim=1)
        self.error = self.R**2 + torch.mean((distance - self.R**2) * (distance > self.R**2)) / self.v
        self.error *= self.alpha
        return code
    def decode(self, x):
        return self.decoding_layer(x)

    def insidePercentage(self, x):
        code = self.encode(x)
        return torch.sum(torch.sum((code - self.center.expand(code.shape[0], code.shape[1])) ** 2, dim=1) <= self.R**2) / len(code)

    def forward(self, x):
        return self.decode(self.encode(x))

class DCOCAE(AutoEncoder):
    def __init__(self, num_feature, output_feature=0, center:torch.Tensor=torch.tensor([[0]]), v:float=1e-1, R1:torch.Tensor=torch.tensor(1, dtype=float, requires_grad=True), R2:torch.Tensor=torch.tensor(1, dtype=float, requires_grad=True), latent_dim=4, hidden_dim=8, activation='leaky_relu', initialization='xavier_normal', **kwargs) -> None:
        super().__init__(num_feature=num_feature, output_feature=output_feature, hidden_dim=hidden_dim, latent_dim=latent_dim, activation=activation, initialization=initialization)
        if not 0 < v <= 1:
            raise ValueError('Tradeoff v should be in range (0, 1]')
        self.name = 'DCOCAE'
        self.center, self.v = center, v
        self.R1, self.R2 = nn.Parameter(R1), R2
        self.error = 0

    def encode(self, x, y=None):
        if y:
            x, y = self.encoding_layer(x), self.encoding_layer(y)
            distance_x = torch.sum((x - self.center.expand(x.shape[0], x.shape[1])) ** 2, dim=1)
            distance_y = torch.sum((y - self.center.expand(y.shape[0], y.shape[1])) ** 2, dim=1)
            error_x = (distance_x - self.R1**2) * (distance_x > self.R1**2) + (self.R2**2 - distance_x) * (distance_x < self.R2**2)
            error_y = ((self.R1 - distance_y)**2 + (distance_y - self.R2)**2) * torch.logical_and(distance_y < self.R1**2, distance_y > self.R2**2)
            self.error = (self.R1 - self.R2)**2 + (torch.sum(error_x) + torch.sum(error_y)) / (len(x) + len(y))  / self.v
            return x, y
        x = self.encoding_layer(x)
        distance_x = torch.sum((x - self.center.expand(x.shape[0], x.shape[1])) ** 2, dim=1)
        error_x = (distance_x - self.R1**2) * (distance_x > self.R1**2) + (self.R2**2 - distance_x) * (distance_x < self.R2**2)
        self.error = (self.R1 - self.R2)**2 + torch.mean(error_x)  / self.v
        return x
    
    def decode(self, x, y=None):
        if y:
            return self.decoding_layer(x), self.decoding_layer(y)
        return self.decoding_layer(x)

    def insidePercentage(self, x):
        code = self.encode(x)
        distance = torch.sum((code - self.center.expand(code.shape[0], code.shape[1])) ** 2, dim=1)
        return torch.sum(torch.logical_and(self.R2**2 <= distance, distance <= self.R1**2)) / len(code)
    
    def forward(self, x, y=None):
        return self.decode(self.encode(x, y)) 

class DSVDD(nn.Module):
    def _init_weights(self, module):
        if self.initialization == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        elif self.initialization == 'xavier_normal':
            torch.nn.init.xavier_normal_(module.weight, gain=torch.nn.init.calculate_gain(nonlinearity='linear'))

        if module.bias is not None:
            module.bias.data.zero_()

    def __init__(self, num_feature, output_feature=0, hidden_dim=64, center:torch.Tensor=torch.tensor([[0]]), R:torch.Tensor=torch.tensor(1, dtype=float, requires_grad=True), v=1, gamma=1e-1, activation='leaky_relu', initialization='xavier_normal', **kwargs) -> None:
        super().__init__()
        self.name = 'DSVDD'
        if not 0 < v <= 100:
            raise ValueError('v must be in range (0, 1]')
        self.num_feature = num_feature
        self.output_feature = num_feature if output_feature == 0 else output_feature
        self.activation, self.initialization, self.center, self.v, self.gamma = activation, initialization, center, v, gamma
        self.R = nn.Parameter(torch.tensor(1, dtype=float, requires_grad=True))
        self.hd = hd = hidden_dim
        self.l1 = nn.Linear(num_feature, hd, False)
        self.l2 = nn.Linear(hd, hd, False)
        self.l3 = nn.Linear(hd, hd, False)
        self.l4 = nn.Linear(hd, hd, False)
        self.l5 = nn.Linear(hd, hd, False)
        self.l6 = nn.Linear(hd, self.output_feature, False)
        self.error = 0
        for layer in self.children():
            self._init_weights(layer)

        if activation == 'leaky_relu':
            self.kernel = nn.Sequential(
                self.l1,
                nn.LeakyReLU(0.1),
                self.l2,
                nn.LeakyReLU(0.1),
                self.l3,
                nn.LeakyReLU(0.1),
                self.l4,
                nn.LeakyReLU(0.1),
                self.l5,
                nn.LeakyReLU(0.1),
                self.l6,
            )  
        elif activation == 'tanh':
            self.kernel = nn.Sequential(
                self.l1,
                nn.Tanh(),
                self.l2,
                nn.Tanh(),
                self.l3,
                nn.Tanh(),
                self.l4,
                nn.Tanh(),
                self.l5,
                nn.Tanh(),
                self.l6,
            )  
        elif activation == 'sigmoid':
            self.kernel = nn.Sequential(
                self.l1,
                nn.Sigmoid(),
                self.l2,
                nn.Sigmoid(),
                self.l3,
                nn.Sigmoid(),
                self.l4,
                nn.Sigmoid(),
                self.l5,
                nn.Sigmoid(),
                self.l6,
            )
        else: # for now, use leaky relu as default
            self.kernel = nn.Sequential(
                self.l1,
                nn.LeakyReLU(0.1),
                self.l2,
                nn.LeakyReLU(0.1),
                self.l3,
                nn.LeakyReLU(0.1),
                self.l4,
                nn.LeakyReLU(0.1),
                self.l5,
                nn.LeakyReLU(0.1),
                self.l6,
            )  

    def forward(self, x, y=None):
        l2, l2_reg_loss = nn.MSELoss(reduction='mean'), 0
        if y is None:
            x = self.kernel(x)
            distance = torch.sum((x - self.center.expand(x.shape[0], x.shape[1])) ** 2, dim=1)
            self.error = self.R**2 + torch.sum((distance - self.R**2) * (distance > self.R**2)) / self.v
            for param in self.parameters():
                if param is not self.R:
                    l2_reg_loss += 1/l2(param, torch.zeros_like(param))
            self.error += l2_reg_loss * self.gamma
            return x
        x, y = self.kernel(x), self.kernel(y)
        distance_x = torch.sum((x - self.center.expand(x.shape[0], x.shape[1])) ** 2, dim=1)
        distance_y = torch.sum((y - self.center.expand(y.shape[0], y.shape[1])) ** 2, dim=1)
        error_y = (distance_y < self.R**2) * (self.R**2 - distance_y)
        self.error = self.R**2 + (torch.sum((distance_x - self.R**2) * (distance_x > self.R**2)) + torch.sum(error_y)) / (len(x) + len(y))  / self.v
        for param in self.parameters():
            l2_reg_loss += 1/l2(param, torch.zeros_like(param))
        self.error += l2_reg_loss * self.gamma
        return x, y
    
    def insidePercentage(self, x):
        code = self.forward(x)
        return torch.sum(torch.sum((code - self.center.expand(code.shape)) ** 2, dim=1) <= self.R**2) / len(code)

class DSVDE(nn.Module):
    def _init_weights(self, module):
        if self.initialization == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        elif self.initialization == 'xavier_normal':
            torch.nn.init.xavier_normal_(module.weight, gain=torch.nn.init.calculate_gain(nonlinearity='linear'))
        if module.bias is not None:
            module.bias.data.zero_()

    def __init__(self, num_feature, output_feature=0, hidden_dim=64, center:torch.Tensor=torch.tensor([[0]]), R1:torch.Tensor=torch.tensor(1, dtype=float, requires_grad=True), R2:torch.Tensor=torch.tensor(1, dtype=float, requires_grad=True), v=1, activation='leaky_relu', initialization='xavier_normal', **kwargs) -> None:
        super().__init__()
        self.name = 'DSVDE'
        if not 0 < v <= 1:
            raise ValueError('v must be in range (0, 1]')
        self.num_feature = num_feature
        self.output_feature = num_feature if output_feature == 0 else output_feature
        self.activation, self.initialization, self.center, self.v = activation, initialization, center, v
        self.R1, self.R2 = nn.Parameter(R1), R2
        self.hd = hd = hidden_dim
        self.l1 = nn.Linear(num_feature, hd, False)
        self.l2 = nn.Linear(hd, hd, False)
        self.l3 = nn.Linear(hd, hd, False)
        self.l4 = nn.Linear(hd, hd, False)
        self.l5 = nn.Linear(hd, hd, False)
        self.l6 = nn.Linear(hd, self.output_feature, False)
        self.error = 0
        for layer in self.children():
            self._init_weights(layer)

        if activation == 'leaky_relu':
            self.kernel = nn.Sequential(
                self.l1,
                nn.LeakyReLU(0.1),
                self.l2,
                nn.LeakyReLU(0.1),
                self.l3,
                nn.LeakyReLU(0.1),
                self.l4,
                nn.LeakyReLU(0.1),
                self.l5,
                nn.LeakyReLU(0.1),
                self.l6,
            )  
        elif activation == 'tanh':
            self.kernel = nn.Sequential(
                self.l1,
                nn.Tanh(),
                self.l2,
                nn.Tanh(),
                self.l3,
                nn.Tanh(),
                self.l4,
                nn.Tanh(),
                self.l5,
                nn.Tanh(),
                self.l6,
            )  
        elif activation == 'sigmoid':
            self.kernel = nn.Sequential(
                self.l1,
                nn.Sigmoid(),
                self.l2,
                nn.Sigmoid(),
                self.l3,
                nn.Sigmoid(),
                self.l4,
                nn.Sigmoid(),
                self.l5,
                nn.Sigmoid(),
                self.l6,
            )
        else: # for now, use leaky relu as default
            self.kernel = nn.Sequential(
                self.l1,
                nn.LeakyReLU(0.1),
                self.l2,
                nn.LeakyReLU(0.1),
                self.l3,
                nn.LeakyReLU(0.1),
                self.l4,
                nn.LeakyReLU(0.1),
                self.l5,
                nn.LeakyReLU(0.1),
                self.l6,
            )  

    def forward(self, x, y=None):
        if y is None:
            x = self.kernel(x)
            distance = torch.sum((x - self.center.expand(x.shape[0], x.shape[1])) ** 2, dim=1)
            self.error = (self.R1 - self.R2)**2 + torch.sum((distance - self.R1**2) * (distance > self.R1**2) + (self.R2**2 - distance) * (distance < self.R2**2)) / self.v
            return x
        x, y = self.kernel(x), self.kernel(y)
        distance_x = torch.sum((x - self.center.expand(x.shape[0], x.shape[1])) ** 2, dim=1)
        distance_y = torch.sum((y - self.center.expand(y.shape[0], y.shape[1])) ** 2, dim=1)
        error_x = (distance_x - self.R1**2) * (distance_x > self.R1**2) + (self.R2**2 - distance_x) * (distance_x < self.R2**2)
        error_y = ((self.R1 - distance_y)**2 + (distance_y - self.R2)**2) * torch.logical_and(distance_y < self.R1**2, distance_y > self.R2**2)
        self.error = (self.R1 - self.R2)**2 + (torch.sum(error_x) + torch.sum(error_y)) / (len(x) + len(y))  / self.v
        return x, y

    def insidePercentage(self, x):
        code = self.forward(x)
        distance = torch.sum((code - self.center.expand(code.shape[0], code.shape[1])) ** 2, dim=1)
        return torch.sum(torch.logical_and(self.R2**2 <= distance, distance <= self.R1**2)) / len(code)

class PEA(nn.Module):
    def _init_weights(self, module):
        if not isinstance(module, nn.Linear):
            return
        """ if issubclass(type(module), nn.modules.activation.LeakyReLU):
            return
        if type(module) == nn.Sequential or type(module) == nn.ModuleList:
            for child in module.children():
                self._init_weights(child) """
            
        if self.initialization == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        elif self.initialization == 'xavier_normal':
            torch.nn.init.xavier_normal_(module.weight, gain=torch.nn.init.calculate_gain(nonlinearity='linear'))
        elif self.initialization == 'ones':
            torch.nn.init.ones_(module.weight)
        elif self.initialization == 'zeros':
            torch.nn.init.zeros_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
        
    def initAllWeights(self):
        for layer in self.children():
            self._init_weights(layer)
        self.sample_weights = (torch.diag(torch.ones(self.sample_size)))
        self.feature_weights = (torch.diag(torch.ones(self.num_feature)))
        
    def initProjector(self, projector):
        for i, seq in enumerate(projector):
            seq.append(nn.Linear(self.num_feature, self.num_feature, bias=True))
            seq.append(nn.LeakyReLU(0.1))
            seq.append(nn.Linear(self.num_feature, self.num_feature, bias=True))
            seq.append(nn.LeakyReLU(0.1))
            seq.append(nn.Linear(self.num_feature, self.latent_dim, bias=True))
    
    def __init__(self, num_feature, sample_size, latent_dim:int=8, num_projections=4, initialization='xavier_normal', **kwargs) -> None:
        super().__init__()
        self.name = 'PEA'
        self.initialization = initialization
        self.num_feature, self.sample_size, self.latent_dim, self.num_projections = num_feature, sample_size, latent_dim, num_projections
        self.noSampleWeights = True
        
        self.l_qf = nn.Linear(num_feature, num_feature)
        self.l_kf = nn.Linear(num_feature, num_feature)
        self.l_vf = nn.Linear(num_feature, num_feature)

        self.projectors_Q = nn.ModuleList([nn.Sequential() for _ in range(self.num_projections)])
        self.projectors_K = nn.ModuleList([nn.Sequential() for _ in range(self.num_projections)])
        #self.projectors_V = nn.ModuleList([nn.Sequential() for _ in range(self.num_projections)])
        self.initProjector(projector=self.projectors_Q)
        self.initProjector(projector=self.projectors_K)
        self.multiHeadFeatureNum = self.num_projections*self.latent_dim
            
        self.initAllWeights()
        
        self.Q_f = nn.Sequential(
            nn.Linear(num_feature, num_feature),
            nn.LeakyReLU(0.1),
            nn.Linear(num_feature, num_feature),
            nn.LeakyReLU(0.1),
            nn.Linear(num_feature, num_feature),
        )
        self.K_f = nn.Sequential(
            nn.Linear(num_feature, num_feature),
            nn.LeakyReLU(0.1),
            nn.Linear(num_feature, num_feature),
            nn.LeakyReLU(0.1),
            nn.Linear(num_feature, num_feature),
            nn.LeakyReLU(0.1),
        )
        self.V_f = nn.Sequential(
            nn.Linear(num_feature, num_feature),
            nn.LeakyReLU(0.1),
            nn.Linear(num_feature, num_feature),
            nn.LeakyReLU(0.1),
            nn.Linear(num_feature, num_feature),
            nn.LeakyReLU(0.1),
        )
        self.FeatureBelief = nn.Sequential(
            nn.Linear(self.multiHeadFeatureNum, self.multiHeadFeatureNum,bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(self.multiHeadFeatureNum, self.multiHeadFeatureNum,bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(self.multiHeadFeatureNum, self.num_feature,bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(self.num_feature, self.num_feature,bias=True),
        )
        self.sm = nn.Softmax(dim=-1)
        
    def getFeatureWeights(self, Q, K):
        #A = torch.softmax(torch.matmul(torch.t(K), Q), dim=-1)
        A = 1 / (torch.matmul(torch.t(K), Q)**2) 
        #A = torch.matmul(torch.t(K), Q)
        attention_score = torch.sum(A, dim=0, keepdim=False)
        #attention_score = attention_score / torch.sum(attention_score)
        attention_score_sm = nn.functional.softmax(attention_score, dim=-1) * len(attention_score)
        #attention_score = nn.functional.softmax(attention_score)
        #attention_score_sm *= len(attention_score_sm) # re-scale the attention score
        return attention_score_sm
    
    def getSampleWeights(self, Q, K):
        #A = torch.softmax(torch.matmul(Q, torch.t(K)), dim=-1)
        A = torch.matmul(Q, torch.t(K))
        #A = A / torch.sum(A, dim=-1, keepdim=True)
        #A = self.sm(A)
        #attention_score = self.sm(torch.sum(A, dim=0, keepdim=False))
        attention_score = torch.sum(A, dim=0, keepdim=False)
        attention_score = nn.functional.normalize(attention_score, dim=-1)
        #attention_score = attention_score / torch.sum(attention_score)
        #attention_score *= len(attention_score) # re-scale the attention score
        return attention_score
    
    def forward(self, x):
        # get feature weights
        feature_weights = torch.diag(self.getFeatureWeights(self.Q_f(x), self.K_f(x)))
        self.feature_weights = feature_weights.detach().clone()
        # get outputs of all the projection networks
        Q, K, V = self.Q_f(x), self.K_f(x), self.V_f(x)
        projections_Q = [seq(Q) for seq in self.projectors_Q]
        projections_K = [seq(K) for seq in self.projectors_K]
        
        if self.noSampleWeights:
            multi_head_feature_weights = torch.empty(0)
            for i in range(self.num_projections):
                p_Q, p_K = projections_Q[i], projections_K[i]
                multi_head_feature_weights = torch.concat((multi_head_feature_weights, self.getFeatureWeights(p_Q, p_K)), dim=-1)
            multi_head_feature_weights_sm = nn.functional.softmax(self.FeatureBelief(multi_head_feature_weights), dim=-1) * self.num_feature
            multi_head_feature_weights_sm = torch.diag(multi_head_feature_weights_sm)
            self.multi_head_feature_weights = multi_head_feature_weights_sm
            return torch.linalg.multi_dot((V, feature_weights, multi_head_feature_weights_sm))
        else:
            # pass them to get sample weights
            sample_weights = torch.zeros(len(x))
            for i in range(self.num_projections):
                p_Q, p_K = projections_Q[i], projections_K[i]
                sample_weights += self.getSampleWeights(p_Q, p_K)
            # get the average of all the sample weights
            sample_weights = sample_weights/self.num_projections
            x = x * sample_weights.view(len(x), 1).expand(x.shape)
            self.sample_weights = sample_weights.detach().clone()
            #return torch.matmul(sample_weights, torch.matmul(x, feature_weights))
            return torch.matmul(x, feature_weights)
    
    def transformData(self, x):
        with torch.no_grad():
            if self.noSampleWeights:
                x = self.V_f(x)
                return torch.linalg.multi_dot((x, self.feature_weights, self.multi_head_feature_weights))
            x_clone = x.clone()
            for i, _ in enumerate(x):
                x_clone[i:i+1] = self.forward(x[i:i+1])
        return x_clone    
    
# Restricted Boltzmann Machine
class RBM(nn.Module):
    def _init_weights(self, module):
        if self.initialization == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        elif self.initialization == 'xavier_normal':
            torch.nn.init.xavier_normal_(module.weight, gain=torch.nn.init.calculate_gain(nonlinearity='linear'))
        elif self.initialization == 'ones':
            torch.nn.init.ones_(module.weight)
        elif self.initialization == 'zeros':
            torch.nn.init.zeros_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
    
    def __init__(self, num_feature, latent_dim:int=8, initialization='xavier_normal', **kwargs) -> None:
        super().__init__()
        self.name = 'RBM'
        self.num_feature, self.latent_dim, self.initialization = num_feature, latent_dim, initialization
        self.W = nn.Linear(num_feature, latent_dim, False)
        self.H = nn.Parameter(torch.rand(1, self.latent_dim))
        self.A, self.B = nn.Linear(num_feature, 1, False), nn.Linear(latent_dim, 1, False)
        # error term
        self.error = None
        # init weights
        for layer in self.children():
            self._init_weights(layer)
            
    def energy(self, x):
        return - self.A(x) - self.B(self.H) - torch.matmul(self.W(x), torch.t(self.H))
        
    def forward(self, x):
        E = self.energy(x)
        P = torch.exp(-E) # n x 1
        P = torch.sigmoid(P)
        self.error = torch.sum(-(P))
        #self.error = -torch.mean(P)
        return P
   
# Stacked VAE
class SVAE(VAE):
    def __init__(self, num_feature, output_feature=0, latent_dim=16, hidden_dim=8, activation='leaky_relu', initialization='xavier_normal', sigma=1, layer_config=[[16, 16, 16, 8], [8, 16, 16, 16]], **kwargs) -> None:
        super().__init__(num_feature=num_feature, output_feature=output_feature, latent_dim=latent_dim, hidden_dim=hidden_dim, activation=activation, initialization=initialization, sigma=sigma, layer_config=layer_config, **kwargs)
        self.name = 'SVAE'
        self.submodule1 = VAE(num_feature=latent_dim, latent_dim=latent_dim, activation='leaky_relu', initialization='xavier_normal', sigma=sigma*1e-2, layer_config=[[8, 8, 8, 8], [8, 8, 8, 8]])
        self.submodule2 = VAE(num_feature=latent_dim, latent_dim=latent_dim, activation='leaky_relu', initialization='xavier_normal', sigma=sigma*1e-4, layer_config=[[8, 8, 8, 4], [4, 8, 8, 8]])
        
    def encode(self, x):
        x = self.encoding_layer(x)
        # distribution generation
        mu, var = self.generate_mean(x), torch.exp(10*nn.Tanh()(self.generate_log_var((x))))
        return self.reparameterize(mu, var)
    
    def encode_all(self, x):
        x = self.encode(x)
        x = self.submodule1.encode(x)
        x = self.submodule2.encode(x)
        self.error += self.submodule1.error + self.submodule2.error
        return x
    
    def decode_all(self, x):
        x = self.submodule2.decode(x)
        x = self.submodule1.decode(x)
        x = self.decode(x)
        return x
    
    def forward(self, x):
        return self.decode_all(self.encode_all(x))
                
# Stacked AE
class SAE(nn.Module):
    def __init__(self, num_feature, output_feature=0, latent_dim=8, hidden_dim=16, activation='leaky_relu', initialization='xavier_normal', layer_config=[[16, 16, 16, 8], [8, 16, 16, 16]], **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = 'SAE'
        self.num_feature, self.latent_dim, self.hidden_dim, self.initialization = num_feature, latent_dim, hidden_dim, initialization # arguments
        self.output_feature = num_feature if output_feature == 0 else output_feature
        self.submodule1 = AutoEncoder(num_feature=num_feature, latent_dim=latent_dim, hidden_dim=hidden_dim, output_feature=output_feature, activation=activation, initialization=initialization, layer_config=layer_config)
        self.submodule2 = AutoEncoder(num_feature=2*self.submodule1.output_feature, latent_dim=latent_dim, hidden_dim=hidden_dim, activation=activation, initialization=initialization, layer_config=layer_config)
        self.submodule3 = AutoEncoder(num_feature=2*self.submodule2.output_feature, output_feature=num_feature, latent_dim=latent_dim*2, hidden_dim=hidden_dim, activation=activation, initialization=initialization, layer_config=layer_config)
        self.submodule_list = []
        self.error = None
        
    def forward(self, x):
        output = self.submodule1(x)
        x = torch.concat((x, output), dim=-1)
        output = self.submodule2(x)
        x = torch.concat((x, output), dim=-1)
        output = self.submodule3(x)
        return output            

# Feature Extracting AE
class FEAE(pyodBase.BaseDetector):
    def __init__(self, contamination:float=0.1, n_heads:int=1, n_kernel_dim:int=16, device:str='cpu'):
        super(FEAE, self).__init__(contamination=contamination)
        try:
            valid = (n_heads//1 == n_heads and n_heads > 0)
            if not valid:
                raise ValueError()
        except:
            raise ValueError("n_heads \'{}\' is not a valid integer".format(n_heads))
        try:
            valid = n_kernel_dim//1 == n_kernel_dim and n_kernel_dim > 0
            if not valid:
                raise ValueError()
        except:
            raise ValueError("n_kernel_dim \'{}\' is not a valid integer".format(n_kernel_dim))
        self.n_heads = n_heads
        self.n_kernel_dim = n_kernel_dim
        self.FC
        self.device = torch.device(device)
        
    # helper function to build multi-head attention nets
    def build_transformer_(self, n_feature, n_kernel_dim, n_heads:int=1, n_layers:int=2):
        self.multi_head_Q, self.multi_head_K = nn.ModuleList([]).to(self.device), nn.ModuleList([]).to(self.device)
        for _ in range(n_heads):
            getQ, getK = [], []
            for i in range(n_layers):
                if i == 0:
                    getQ.extend(nn.Linear(n_feature, n_kernel_dim, bias=False), nn.LeakyReLU(0.1))
                    getK.extend(nn.Linear(n_feature, n_kernel_dim, bias=False), nn.LeakyReLU(0.1))
                    continue
                getQ.extend(nn.Linear(n_kernel_dim, n_kernel_dim, bias=False))
                getK.extend(nn.Linear(n_kernel_dim, n_kernel_dim, bias=False))
                if i == n_layers-1:
                    getQ.extend(nn.LeakyReLU(0.1))
                    getK.extend(nn.LeakyReLU(0.1))
            # append nn.Sequential created from this MLP to the multi-head-attention module list
            self.multi_head_Q.extend(nn.Sequential(*getQ))
            self.multi_head_K.extend(nn.Sequential(*getK))
        self.PEFN = nn.Sequential(
                nn.Linear(n_kernel_dim, n_kernel_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(n_kernel_dim, 1),
            )
        

    def fit(self, X, y=None, n_epochs=1000, lr=1e-3):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)
        """
        
        n_samples, n_features = X.shape[0], X.shape[1]
        # convert to torch tensor
        X = torch.tensor(X).to(torch.float32).to(self.device)
        # init submodules
        self.build_transformer_(n_feature=n_features, n_kernel_dim=self.n_kernel_dim, n_heads=self.n_heads)
        
        # start training
        
            
        
        
        # train
        return self
        
    def set_params(self, **kwargs):
        # check existed attr
        #
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def decision_function(self, X):
        pass
    
    def forward(self, X):
        # attention mechanism
        attentions = torch.empty(0)
        for getQ, getK in zip(self.multi_head_Q, self.multi_head_K):
            att = torch.matmul(torch.t(getQ(X)), getK(X)) # k x k
            att = self.PEFN(att) # k x 1
            attentions = torch.concatenate((attentions, att), dim=0)
            pass

