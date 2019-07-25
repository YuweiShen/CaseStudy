# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 02:52:25 2019

@author: susie
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 19:55:54 2019

@author: susie
"""
import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import argparse

# cuda

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# hyperparameters
BATCH = 32
epochs = 5
lr = 1e-3
LATENT_DIM = 2
encoder_dim = 32
col_num = 16
sample_num = 14382
decoder_dim = encoder_dim
input_dim = 32
#
#parser = argparse.ArgumentParser(description='VAE')
#parser.add_argument('--batch-size', type=int, default=128, metavar='N',
#                    help='input batch size for training (default: 128)')
#parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                    help='number of epochs to train (default: 10)')
#parser.add_argument('--no-cuda', action='store_true', default=False,
#                    help='enables CUDA training')
#parser.add_argument('--seed', type=int, default=1, metavar='S',
#                    help='random seed (default: 1)')
#parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                    help='how many batches to wait before logging training status')
#args = parser.parse_args()
#args.cuda = not args.no_cuda and torch.cuda.is_available()


## 加载数据
train_data = pd.read_csv('C:/Users/58454/Desktop/GAN/data/nltcs_train.csv')
test_data = pd.read_csv('C:/Users/58454/Desktop/GAN/data/nltcs_test.csv')
#train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH, shuffle=True)
dtest = train_data.values
dtest = torch.FloatTensor(dtest)
xtrain = torch.unsqueeze(dtest, dim=1)
x = Variable(xtrain)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv1d(in_channels = 1,out_channels = input_dim,kernel_size = 3 ),
                nn.BatchNorm1d(input_dim),
                nn.ReLU(),
                
                nn.Conv1d(input_dim, encoder_dim, 3),
                nn.BatchNorm1d(encoder_dim),
                nn.ReLU()
                )
        self.fcmu = nn.Linear(encoder_dim*12, LATENT_DIM) #均值,
        self.fcsigma = nn.Linear(encoder_dim*12, LATENT_DIM) #标准差
        self.fc = nn.Linear(LATENT_DIM, encoder_dim * 12)  
        self.decoder = nn.Sequential(              
                nn.ConvTranspose1d(decoder_dim, input_dim, kernel_size=3),
                nn.ReLU(),
                
                nn.ConvTranspose1d(input_dim, 1, kernel_size=3),
                nn.Sigmoid()
                )

    def reparameterize(self,mu,logvar):
        if device.type == 'cpu':
            epsi = Variable(torch.randn(mu.size(0), mu.size(1)))
        elif device.type == 'gpu':
            epsi = Variable(torch.randn(mu.size(0), mu.size(1))).cuda()
        z = mu + epsi*torch.exp(logvar/2)
        return z
    
    def forward(self,x):
        out1, out2 =self.encoder(x), self.encoder(x) # data_size,encoder_dim,12
        mu = self.fcmu(out1.view(out1.size(0),-1)) #data_size, latent_num
        logvar = self.fcsigma(out2.view(out2.size(0),-1)) #data_size, latent_num
        z = self.reparameterize(mu, logvar)#data_size, latent_num
        out3 = self.fc(z).view(z.size(0), encoder_dim, 12) #data_size, encoder_dim*col_num
        #to data_size,encoder_dim,col_num
        return self.decoder(out3), mu, logvar

def loss_func(reconstruct, x, mu, logvar):
     BCE = fun.binary_cross_entropy(reconstruct, x,  size_average=False)
     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
     return BCE+KLD

vae = VAE().to(device)
optimizer =  optim.Adam(vae.parameters(), lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)
 
def vaetrain( path, vae, data, lr, epochs, steps_per_epoch, GPU=  True):
    vae.train()
    for epoch in range(epochs):
        print("----------pretrain Epoch %d:----------\n"%epoch)
        log = open(path+"train_log_vae.txt","a+")
        log.write("----------pretrain Epoch %d:----------\n"%epoch)
        log.close()
        it = 0
        while it < steps_per_epoch:
            dtest = data.values
            dtest = torch.FloatTensor(dtest)
            x = torch.unsqueeze(dtest, dim=1)
            if GPU:
                x = Variable(x).cuda()
            else:
                x = Variable(x)
            optimizer.zero_grad()
            x_, mu, logvar = vae.forward(x)
            loss = loss_func(x_, x, mu, logvar)
            loss.backward()
            optimizer.step()
                
            if it%100 == 0:
                print("VAE iteration {}, loss: {}\n".format(it, loss.data))
                log = open(path+"train_log_vae.txt","a+")
                log.write("VAE iteration {}, loss: {}\n".format(it, loss.data))
                log.close()  
                sample = Variable(torch.randn(sample_num, LATENT_DIM))
                sample = vae.decoder(vae.fc(sample).view(sample_num, encoder_dim, 12))
                sample_data = pd.DataFrame(np.round(sample.detach().numpy().reshape(sample_num, col_num)))
                sample_data.to_csv(path+'sample_data_vae_{}.csv'.format(epoch), index = None)
            it += 1
            if it >= steps_per_epoch:
                break

vaetrain('C:\\Users\\58454\\Desktop\\GAN', vae, train_data, lr, epochs, 5000, False)
