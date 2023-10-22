#!/usr/bin/env python
# coding: utf-8

import yaml
import time
import copy
import joblib
import numpy as np
import pandas as pd
import ase
from ase import Atoms, io, build

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# structrepgen
from structrepgen.utils.dotdict import dotdict
from structrepgen.utils.utils import torch_device_select
from structrepgen.reconstruction.reconstruction import Reconstruction
from structrepgen.descriptors.ensemble import EnsembleDescriptors

torch.set_printoptions(precision=4, sci_mode=False)


class Encoder(nn.Module):
    ''' This the encoder part of VAE
    '''
    def __init__(self, input_dim, hidden_dim, latent_dim, hidden_layers, y_dim, act_type, act_args):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.hidden = nn.Sequential()
        dims_arr = np.linspace(hidden_dim, latent_dim, hidden_layers+1, dtype=int)[0:-1]
        dims_arr = np.concatenate(([input_dim + y_dim], dims_arr))
        
        for i, (in_size, out_size) in enumerate(zip(dims_arr[:-1], dims_arr[1:])):
            self.hidden.add_module(
                name='Linear_'+str(i),
                module=nn.Linear(in_size, out_size)
            )
            self.hidden.add_module(
                name='Act_'+str(i), 
                module=getattr(nn, act_type)(**act_args)
            )

        self.mu = nn.Linear(dims_arr[-1], latent_dim)
        self.var = nn.Linear(dims_arr[-1], latent_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim + n_classes]
        x = self.hidden(x)
        # hidden is of shape [batch_size, hidden_dim]
        # latent parameters
        mean = self.mu(x)
        # mean is of shape [batch_size, latent_dim]
        log_var = self.var(x)
        # log_var is of shape [batch_size, latent_dim]
        return mean, log_var

class Decoder(nn.Module):
    ''' This the decoder part of VAE
    '''

    def __init__(self, latent_dim, hidden_dim, output_dim, hidden_layers, y_dim, act_type, act_args):
        '''
        Args:
            latent_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the size of output.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.hidden = nn.Sequential()
        dims_arr = np.linspace(latent_dim + y_dim, hidden_dim, hidden_layers+1, dtype=int)
        
        for i, (in_size, out_size) in enumerate(zip(dims_arr[:-1], dims_arr[1:])):
            self.hidden.add_module(
                name='Linear_'+str(i),
                module=nn.Linear(in_size, out_size)
            )
            self.hidden.add_module(
                name='Act_'+str(i), 
                module=getattr(nn, act_type)(**act_args)
            )

        self.hidden_to_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim + num_classes]
        x = self.hidden(x)
        generated_x = self.hidden_to_out(x)
        return generated_x


class CVAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.
    '''

    def __init__(self, input_dim, hidden_dim, latent_dim, hidden_layers, y_dim, act_type, act_args={}):
        '''
        Args:
            input_dim: A integer indicating the size of input.
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, hidden_layers, y_dim, act_type, act_args)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, hidden_layers, y_dim, act_type, act_args)

    def forward(self, x, y):

        x = torch.cat((x, y), dim=1)

        # encode
        z_mu, z_var = self.encoder(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = torch.cat((x_sample, y), dim=1)

        # decode
        generated_x = self.decoder(z)

        return generated_x, z_mu, z_var

def kl_divergence(z, mu, std):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)

    # sum over last dim to go from single dim distribution to multi-dim
    kl = kl.sum(-1)
    return kl


def calculate_loss(x, reconstructed_x, mu, log_var, weight, mc_kl_loss):
    # reconstruction loss
    rcl = F.mse_loss(reconstructed_x, x,  reduction='mean')
    # kl divergence loss

    if mc_kl_loss == True:
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        kld = kl_divergence(z, mu, std).sum()
    else:
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        #kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    rcl=rcl*10000
    kld=kld*10000
    #print(rcl, kld)
    
    return rcl, kld * weight


# ### CONFIG

# In[38]:


CONFIG = {}

CONFIG['gpu'] = True
CONFIG['params'] = {}
CONFIG['params']['seed'] = 42
CONFIG['params']['split_ratio'] = 0.2
CONFIG['params']['input_dim'] = 708
CONFIG['params']['hidden_dim'] = 512
CONFIG['params']['latent_dim'] = 64
CONFIG['params']['hidden_layers'] = 3
CONFIG['params']['y_dim'] = 1
CONFIG['params']['batch_size'] = 512
CONFIG['params']['n_epochs'] = 1500
CONFIG['params']['lr'] = 5e-4
CONFIG['params']['final_decay'] = 0.2
CONFIG['params']['weight_decay'] = 0.001
CONFIG['params']['verbosity'] = 10
CONFIG['params']['kl_weight'] = 0.001
CONFIG['params']['mc_kl_loss'] = False
CONFIG['params']['act_fn'] = "ELU"

CONFIG['data_x_path'] = 'MP_data_csv/mp_20_raw_train_dist_mat.pt'
CONFIG['data_ead_path'] = 'MP_data_csv/mp_20_raw_train_ead_mat.pt'
CONFIG['composition_path'] = 'MP_data_csv/mp_20_raw_train_composition_mat.pt'
CONFIG['cell_path'] = 'MP_data_csv/mp_20_raw_train_cell_mat.pt'
CONFIG['data_y_path'] = "MP_data_csv/mp_20/raw_train/targets.csv"
CONFIG['unprocessed_path'] = 'MP_data_csv/mp_20_raw_train_unprocessed.txt'
CONFIG['model_path'] = 'saved_models/cvae_saved.pt'
CONFIG['model_path2'] = 'saved_models/cvae_saved_dict.pt'
CONFIG['scaler_path'] = 'saved_models/scaler.gz'

CONFIG['train_model'] = False
CONFIG['generate_samps'] = 5

CONFIG = dotdict(CONFIG)
CONFIG['descriptor'] = ['ead', 'distance']
# EAD params
CONFIG['L'] = 1
CONFIG['eta'] = [1, 20, 90]
CONFIG['Rs'] = np.arange(0, 10, step=0.2)
CONFIG['derivative'] = False
CONFIG['stress'] = False

CONFIG['all_neighbors'] = True
CONFIG['perturb'] = False
CONFIG['load_pos'] = False
CONFIG['cutoff'] = 20.0
CONFIG['offset_count'] = 3

# ### Trainer


class Trainer():
    def __init__(self, CONFIG) -> None:
        self.CONFIG = CONFIG

        # check GPU availability & set device
        self.device = torch_device_select(self.CONFIG.gpu)

        # initialize
        self.create_data()
        self.initialize()

    def create_data(self):
        p = self.CONFIG.params
        
        unprocessed = set()
        with open(self.CONFIG.unprocessed_path, 'r') as f:
            for l in f.readlines():
                unprocessed.add(int(l))

        # load pt files
        dist_mat = torch.load(self.CONFIG.data_x_path).to("cpu")
        ead_mat = torch.load(self.CONFIG.data_ead_path).to("cpu")
        composition_mat = torch.load(self.CONFIG.composition_path).to("cpu")
        cell_mat = torch.load(self.CONFIG.cell_path).to("cpu")

        # build index
        _ind = [i for i in range(dist_mat.shape[0]) if i not in unprocessed]
        indices = torch.tensor(_ind, dtype=torch.long).to("cpu")

        # select rows
        dist_mat = dist_mat[indices]
        ead_mat = ead_mat[indices]
        composition_mat = composition_mat[indices]
        cell_mat = cell_mat[indices]
        
        # normalize composition
        sums = torch.sum(composition_mat, axis=1).view(-1,1)
        composition_mat = composition_mat / sums
        composition_mat = torch.cat((composition_mat, sums), dim=1)
        
        print(composition_mat.shape, cell_mat.shape, dist_mat.shape, ead_mat.shape)
        data_x = torch.cat((ead_mat/1000000, dist_mat, composition_mat, cell_mat), dim=1)        
        print(data_x.shape)
        
        # scale
        # scaler = MinMaxScaler()
        # data_x = scaler.fit_transform(data_x)
        # # data_x = torch.tensor(data_x, dtype=torch.float)
        # joblib.dump(scaler, self.CONFIG.scaler_path)

        # get y
        y = []
        with open(self.CONFIG.data_y_path, 'r') as f:
            for i, d in enumerate(f.readlines()):
                if i not in unprocessed:
                    y.append(float(d.split(',')[1]))

        data_y = np.reshape(np.array(y), (-1,1))
        print(data_y.shape)

        # train/test split and create torch dataloader
        xtrain, xtest, ytrain, ytest = train_test_split(
            data_x, data_y, test_size=self.CONFIG.split_ratio, random_state=p.seed
        )
        if not isinstance(xtrain, torch.Tensor):
            self.x_train = torch.tensor(xtrain, dtype=torch.float)
        else:
            self.x_train = xtrain
            
        if not isinstance(ytrain, torch.Tensor):
            self.y_train = torch.tensor(ytrain, dtype=torch.float)
        else:
            self.y_train = ytrain
            
        if not isinstance(xtest, torch.Tensor):
            self.x_test = torch.tensor(xtest, dtype=torch.float)
        else:
            self.x_test = xtest
            
        if not isinstance(ytest, torch.Tensor):
            self.y_test = torch.tensor(ytest, dtype=torch.float)
        else:
            self.y_test = ytest
        

        indices = ~torch.any(self.x_train.isnan(),dim=1)
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]
        indices = ~torch.any(self.x_train[:,:601] > 10 ,dim=1)
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]     
        print(self.x_train.shape, self.y_train.shape)   
        print(self.x_train.max())
        
        
        indices = ~torch.any(self.x_test.isnan(),dim=1)
        self.x_test = self.x_test[indices]
        self.y_test = self.y_test[indices]
        indices = ~torch.any(self.x_test[:,:601] > 10 ,dim=1)
        self.x_test = self.x_test[indices]
        self.y_test = self.y_test[indices]        
  


        self.train_loader = DataLoader(
            TensorDataset(self.x_train, self.y_train),
            batch_size=p.batch_size, shuffle=True, drop_last=False
        )

        self.test_loader = DataLoader(
            TensorDataset(self.x_test, self.y_test),
            batch_size=p.batch_size, shuffle=False, drop_last=False
        )
    
    def initialize(self):
        p = self.CONFIG.params

        # create model
        self.model = CVAE(p.input_dim, p.hidden_dim, p.latent_dim, p.hidden_layers, p.y_dim, p.act_fn)
        self.model.to(self.device)
        print(self.model)

        # set up optimizer
        # gamma = (p.final_decay)**(1./p.n_epochs)
        scheduler_args = {"mode":"min", "factor":0.8, "patience":20, "min_lr":1e-7, "threshold":0.0001}
        self.optimizer = optim.AdamW(self.model.parameters(), lr=p.lr, weight_decay=p.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **scheduler_args
        )

    def train(self):
        p = self.CONFIG.params
        self.model.train()

        # loss of the peoch
        rcl_loss = 0.
        kld_loss = 0.

        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # forward
            reconstructed_x, z_mu, z_var = self.model(x, y)

            rcl, kld = calculate_loss(x, reconstructed_x, z_mu, z_var, p.kl_weight, p.mc_kl_loss)

            # backward
            combined_loss = rcl + kld
            combined_loss.backward()
            rcl_loss += rcl.item()
            kld_loss += kld.item()

            # update the weights
            self.optimizer.step()
        
        return rcl_loss, kld_loss
    
    def test(self):
        p = self.CONFIG.params

        self.model.eval()

        # loss of the evaluation
        rcl_loss = 0.
        kld_loss = 0.

        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                # forward pass
                reconstructed_x, z_mu, z_var = self.model(x, y)

                # loss
                rcl, kld = calculate_loss(x, reconstructed_x, z_mu, z_var, p.kl_weight, p.mc_kl_loss)
                rcl_loss += rcl.item()
                kld_loss += kld.item()
        
        return rcl_loss, kld_loss

    def run(self):
        p = self.CONFIG.params
        best_test_loss = float('inf')
        best_train_loss = float('inf')
        best_epoch = 0

        for e in range(p.n_epochs):
            tic = time.time()

            rcl_train_loss, kld_train_loss = self.train()
            rcl_test_loss, kld_test_loss = self.test()

            rcl_train_loss /= len(self.x_train)
            kld_train_loss /= len(self.x_train)
            train_loss = rcl_train_loss + kld_train_loss
            rcl_test_loss /= len(self.x_test)
            kld_test_loss /= len(self.x_test)
            test_loss = rcl_test_loss + kld_test_loss

            self.scheduler.step(train_loss)
            lr = self.scheduler.optimizer.param_groups[0]["lr"]

            if best_test_loss > test_loss:
                best_epoch = e
                best_test_loss = test_loss
                best_train_loss = train_loss
                model_best = copy.deepcopy(self.model)
        
                torch.save(model_best, self.CONFIG.model_path)
                torch.save(model_best.state_dict(), self.CONFIG.model_path2)                
            
            elapsed_time = time.time() - tic
            epoch_out = f'Epoch {e:04d}, Train RCL: {rcl_train_loss:.5f}, Train KLD: {kld_train_loss:.5f}, Train: {train_loss:.5f}, Test RLC: {rcl_test_loss:.5f}, Test KLD: {kld_test_loss:.5f}, Test: {test_loss:.5f}, LR: {lr:.5f}, Time/Epoch (s): {elapsed_time:.3f}'
            if e % p.verbosity == 0:
                print(epoch_out)

        return best_epoch, best_train_loss, best_test_loss



trainer = Trainer(CONFIG)
print(trainer.x_train.shape, trainer.x_test.shape)
#print(trainer.x_train[0], trainer.x_train.isnan(), trainer.x_train.max())

if CONFIG.train_model == True:
    trainer.run()
    
model = trainer.model
model.load_state_dict(torch.load(CONFIG.model_path2))
model.eval()
out = model(trainer.x_test[:10].to(trainer.device), trainer.y_test[:10].to(trainer.device))
print(out[0].shape, trainer.x_test[0:10].shape)


#print("1", out[0][:,-7:])
#print("2", trainer.x_test[0:10, -7:])
#print("3", trainer.y_test[0:50])
#print("4", out[0][0, 0:107])
#print("5", out[0][0, -107:])
#print("6", trainer.x_test[0,:107])
#print("7", trainer.x_test[0, -107:])

###Add for loop to generate n samples
data_list=[]
for i in range(0, CONFIG.generate_samps):
    z = torch.randn(1, CONFIG.params.latent_dim).to(trainer.device)
    y = torch.tensor([[-1.5]]).to(trainer.device)
    z = torch.cat((z, y), dim=1)
    reconstructed_x = model.decoder(z)
    #print(reconstructed_x.shape, reconstructed_x[0,-107:].shape, reconstructed_x[0,:107], reconstructed_x[0,-107:])
    atomic_number_vector = torch.round(reconstructed_x[0,-107:-7]*reconstructed_x[0,-7]).int()
    #print(reconstructed_x[0,-107:-7]*reconstructed_x[0,-7], atomic_number_vector)
    atomic_numbers = []
    for j in range(0, 100):
        atomic_numbers.append(atomic_number_vector[j]*[str(j+1)])
    atomic_numbers = [item for sublist in atomic_numbers for item in sublist]
    cell = reconstructed_x[0,-6:]
    cell[3:6] = cell[3:6]*180/np.pi
    placeholder = Atoms(numbers=[1], positions=[[0,0,0]], cell=cell.detach().cpu().numpy())
    cell = placeholder.get_cell()
    print(str(i), atomic_numbers, len(atomic_numbers), reconstructed_x[0,-7])
    
    data = {}
    data['positions'] = []
    data['atomic_numbers'] = np.array(atomic_numbers, dtype="int")
    data['cell'] = torch.tensor(np.array(cell), dtype=torch.float, device=torch_device_select(CONFIG.gpu))
    data['representation'] = torch.unsqueeze(reconstructed_x[0,:601], 0).detach()
    #print(data['representation'].shape)
    data_list.append(data)


### add for loop to reconstruct n samples
for i, data in enumerate(data_list):
    if len(data['atomic_numbers']) != 0:
        descriptor = EnsembleDescriptors(CONFIG)
        constructor = Reconstruction(CONFIG)
        best_positions, _ = constructor.basin_hopping(
            data,
            total_trials=3, 
            max_hops=1, 
            lr=0.05, 
            displacement_factor=2, 
            max_loss=0.00001,
            max_iter=100,
            verbose=True
        )
        
        optimized_structure = Atoms(
            numbers=data['atomic_numbers'], positions=best_positions.detach().cpu().numpy(),
            cell=data['cell'].cpu().numpy(),
            pbc=(True, True, True)
        )
        
        supercell = ase.build.make_supercell(optimized_structure, [[3,0,0],[0,3,0],[0,0,3]])
        fname = str(i) + '_recon_supercell_step0_{}.cif'.format(str(constructor.descriptor))
        ase.io.write(fname, supercell)
        
        fname = str(i) + '_recon_step0_{}.cif'.format(str(constructor.descriptor))
        ase.io.write(fname, optimized_structure)
        
        features = descriptor.get_features(best_positions.detach().cpu().numpy(), data['cell'], data['atomic_numbers'])
        np.savetxt(str(i) + '_recon0.csv', features.cpu().numpy(), delimiter=',')
        np.savetxt(str(i) + '_0.csv', data['representation'].cpu().numpy().T, delimiter=',')
