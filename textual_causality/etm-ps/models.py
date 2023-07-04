import torch
import torch.nn.functional as F 
from torch import nn
from torch.nn.modules.activation import ReLU, Sigmoid
from torch.nn.modules.batchnorm import BatchNorm1d

class PSNet(nn.Module):
    def __init__(self, n_features, n_hidden_1=1000, n_hidden_2=300):
        super(PSNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden_1), 
            nn.BatchNorm1d(n_hidden_1), 
            nn.LeakyReLU(), 
            nn.Linear(n_hidden_1, n_hidden_2), 
            nn.BatchNorm1d(n_hidden_2), 
            nn.LeakyReLU(), 
            nn.Linear(n_hidden_2, 1), 
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

class ETM(nn.Module):
    def __init__(self, num_topics, vocab_size, embed_size, vi_nn_hidden_size, 
                    theta_act, embeddings=None, enc_drop=0.5):
        super(ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.vi_nn_hidden_size = vi_nn_hidden_size
        self.enc_drop = enc_drop
        self.nn_drop = nn.Dropout(enc_drop)
        self.theta_act = self.get_activation(theta_act)
        
        ## define the word embedding matrix \rho
        if embeddings is None:
            self.rho = nn.Linear(embed_size, vocab_size, bias=False)  # self.rho.weight: (vocab_size x embed_size)
        else:
            self.rho = nn.Parameter(embeddings.clone().float(), requires_grad=False)

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(embed_size, num_topics, bias=False) #nn.Parameter(torch.randn(embed_size, num_topics))
    
        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
                nn.Linear(vocab_size, vi_nn_hidden_size, bias=True), 
                self.theta_act,
                nn.Linear(vi_nn_hidden_size, vi_nn_hidden_size, bias=True),
                self.theta_act,
            )
        self.mu_q_theta = nn.Linear(vi_nn_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(vi_nn_hidden_size, num_topics, bias=True)  # log of variance

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act 

    def forward(self, normalized_bows=None, theta=None, beta_only=False):
        
        beta = self.get_beta()  # (K x vocab_size)
        if beta_only:   
            return beta

        if theta is None:
            theta, kld_theta = self.get_theta(normalized_bows)
        else:
            kld_theta = None

        preds = self.decode(theta, beta)  # (batch_size x vocab_size)
        topic_feature = torch.mm(theta, self.alphas.weight) # (batch_size x embed_size)
        
        return beta, theta, kld_theta, preds, topic_feature

    def get_beta(self):
        try:
            logit = self.alphas(self.rho.weight)  # (vocab_size x K)   # torch.mm(self.rho, self.alphas)
        except:
            logit = self.alphas(self.rho)
        beta = F.softmax(logit, dim=0).transpose(1, 0)  
        return beta  # (K x vocab_size) 

    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta) # (batch_size x K)
        theta = F.softmax(z, dim=-1)  # (batch_size, K)
        return theta, kld_theta

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)  # (batch_size x vi_nn_hidden_size)
        if self.enc_drop > 0:
            q_theta = self.nn_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta) # (batch_size x K)
        logsigma_theta = self.logsigma_q_theta(q_theta)  # (batch_size x K)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) # (batch_size x K)
            eps = torch.randn_like(std)  # N(0, 1) of dimension (batch_size x K) 
            return eps.mul_(std).add_(mu)  # will do elementwise multiplication and addition
        else:
            return mu

    def decode(self, theta, beta):
        res = torch.mm(theta, beta)  # (batch_size x vocab_size)
        preds = torch.log(res+1e-6)
        return preds 



