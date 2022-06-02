from bipartite import get_cora, bmatching_diverse, get_qpt_matrices
import os
import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchmetrics.functional import auc
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np

def make_cora_net(n_features=2866, n_hidden=200, n_layers=2, n_targets=1):
    if n_layers ==1:
        return nn.Sequential(nn.Linear(n_features, n_targets), nn.Sigmoid())
    else:
        layers = []
        # input layer
        layers.append(nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU()
            ))
        # hidden layers
        for _ in range(n_layers -2) :
            layers.append(nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU()
            ))
        # output layer
        layers.append(nn.Sequential(
            nn.Linear(n_hidden, n_targets),
            nn.Sigmoid()
        ))
        return nn.Sequential(*layers)

solver = bmatching_diverse
objective_fun=lambda x,v,**param: x @ v



    

def batch_solve(param,y,m,relaxation =False):

    sol = []
    for i in range(len(y)):
        sol.append(  solver(y[i], m[i], relaxation=relaxation,**param) )
    return np.array(sol)
def regret_fn(y_hat,y,sol_true,m,param,minimize=False):
    mm = 1 if minimize else -1    
    sol_hat = torch.from_numpy(batch_solve(param, y_hat.detach().numpy(),m.numpy()))
    return ((mm*(sol_hat - sol_true)*y).sum(1)/(sol_true*y).sum(1)).mean()

def regret_aslist(y_hat,y,sol_true,m,param,minimize=False): 
    mm = 1 if minimize else -1    
    sol_hat = torch.from_numpy(batch_solve(param, y_hat.detach().numpy(),m.numpy()))
    return ((mm*(sol_hat - sol_true)*y).sum(1)/(sol_true*y).sum(1))

class twostage_regression(pl.LightningModule):
    def __init__(self,param, lr=1e-1, seed=2, max_epochs=50):
        super().__init__()
        pl.seed_everything(seed)
        self.param = param
        self.net = make_cora_net(n_layers=2)
        self.lr = lr
        self.max_epochs= max_epochs
        self.save_hyperparameters("lr")
    def forward(self,x):
        return self.net(x) 
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(y_hat,y)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction='mean')
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        val_loss= regret_fn(y_hat,y, sol,m,self.param)
        mseloss = criterion(y_hat, y)
        # aucloss = auc(y,y_hat,reorder=True)
        aucloss = torch.tensor([auc(y[i],y_hat[i],reorder=True) for i in range(len(y))]).mean()
        self.log("val_regret", val_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("val_auc", aucloss, prog_bar=True, on_step=True, on_epoch=True, )
       
        return  {"val_regret": val_loss, "val_mse": mseloss,"val_auc":aucloss}
    def validation_epoch_end(self, outputs):
        avg_regret = torch.stack([x["val_regret"] for x in outputs]).mean()
        avg_mse = torch.stack([x["val_mse"] for x in outputs]).mean()
        
        self.log("ptl/val_regret", avg_regret)
        self.log("ptl/val_mse", avg_mse)
        # self.log("ptl/val_accuracy", avg_acc)
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        num_batches = len(self.train_dataloader()) // self.trainer.accumulate_grad_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.lr, epochs=self.max_epochs,
        steps_per_epoch = num_batches)
        return [optimizer], [scheduler]



def spograd(y,sol,m, param,minimize= False):
    mm = 1 if minimize else -1
    class spograd_cls(torch.autograd.Function):

        @staticmethod
        def forward(ctx, y_hat):

            ctx.save_for_backward(y_hat)
            return mm*((y_hat-y)*sol).sum()

        @staticmethod
        def backward(ctx, grad_output):
            y_hat, = ctx.saved_tensors
            y_spo = 2*y_hat - y
            sol_spo = torch.from_numpy(batch_solve(param, y_spo.detach().numpy(),m.numpy()))  
            return (sol- sol_spo)*mm
    return spograd_cls.apply

def bbgrad(y,sol,m, param,mu,minimize=True):
    mm = 1 if minimize else -1
    class bbgrad_cls(torch.autograd.Function):

        @staticmethod
        def forward(ctx, y_hat):
            y_perturbed = (y_hat + mu* y)

            ctx.save_for_backward(y_hat, y_perturbed)
            return mm*((y_hat-y)*sol).sum()

        @staticmethod
        def backward(ctx, grad_output):
            y_hat,y_perturbed = ctx.saved_tensors
            sol_hat =  torch.from_numpy(batch_solve(param, y_hat.detach().numpy(),m.numpy())) 
            sol_perturbed = torch.from_numpy(batch_solve(param, y_perturbed.detach().numpy(),m.numpy()))
            return -mm*(sol_hat - sol_perturbed)/mu 
    return bbgrad_cls.apply



class SPO(twostage_regression):
    def __init__(self, param, lr=1e-1, seed=2, max_epochs=50):
        super().__init__(param, lr, seed, max_epochs)
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        loss = spograd(y,sol,m,self.param)(y_hat)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class Blackbox(twostage_regression):
    def __init__(self,param, lr=1e-1, mu=0.1, seed=2, max_epochs=50):
        super().__init__(param, lr, seed, max_epochs)
        self.mu = mu
        # self.automatic_optimization = False
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        loss = bbgrad(y,sol,m, self.param,self.mu)(y_hat)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss


def MAP(sol,y,cache,minimize=False):
    '''
    sol, y and y_hat are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    '''
    mm = 1 if minimize else -1 
    loss = 0
    # print("shape check", sol.shape, y.shape,y_hat.shape, cache.shape)
    for ii in range(len(y)):
        loss += torch.max(((sol[ii] - cache )*(mm*y[ii]  )).sum(dim=1))
    return loss
def MAP_c(y_hat,y_true,sol,cache,*wd,**kwd):
    y = y_hat 
    return MAP(sol,y,cache)
def MAP_hatc_c(y_hat,y_true,sol,cache,*wd,**kwd):
    y = y_hat - y_true
    return MAP(sol,y,cache)

def NCE(sol,y,cache,minimize=False):
    '''
    sol, y and y_hat are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    '''
    mm = 1 if minimize else -1 
    loss = 0
    # print("shape check", sol.shape, y.shape,y_hat.shape, cache.shape)
    for ii in range(len(y)):
        loss += torch.mean(((sol[ii] - cache )*(mm*y[ii]  )).sum(dim=1))
    return loss
def NCE_c(y_hat,y_true,sol,cache,*wd,**kwd):
    y = y_hat 
    return NCE(sol,y,cache)
def NCE_hatc_c(y_hat,y_true,sol,cache,*wd,**kwd):
    y = y_hat - y_true
    return NCE(sol,y,cache)




def pointwise_loss(y_hat,y_true,sol,cache,*wd,**kwd):
    '''
    sol, y and y_hat are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    f(y_hat,s) is regresson on f(y,s)
    '''
    loss = (torch.matmul(y_hat,cache.transpose(0,1))- torch.matmul(y_true,cache.transpose(0,1))).square().mean()

    return loss

def pairwise_loss(y_hat,y_true,sol,cache,tau=0, minimize=False,mode= 'B'):
    '''
    sol, y and y_hat are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    '''
    mm = 1 if minimize else -1 
    loss = 0
    relu = nn.ReLU()
    for ii in range(len(y_true)):
        _,indices= np.unique((mm*y_true[ii]*cache).sum(dim=1).detach().numpy(),return_index=True)
        ## return indices after sorting the array in ascending order
        if mode == 'B':
            big_ind = [indices[0] for p in range(len(indices)-1)] #good one
            small_ind = [indices[p+1] for p in range(len(indices)-1)] #bad one
        if mode == 'W':
            big_ind = [indices[p] for p in range(len(indices)-1)] #good one
            small_ind = [indices[-1] for p in range(len(indices)-1)] #bad one
        if mode == 'S':
            big_ind = [indices[p] for p in range(len(indices)-1)] #good one
            small_ind = [indices[p+1] for p in range(len(indices)-1)] #bad one


        loss += relu(tau+ mm*( torch.matmul(cache[big_ind], y_hat[ii]) - torch.matmul(cache[small_ind], y_hat[ii])) ).mean()
        
    return loss

def pairwise_diffloss(y_hat,y_true,sol ,cache, minimize=False,mode= 'B',*wd,**kwd):
    '''
    sol, y and y_hat are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    '''
    mm = 1 if minimize else -1 
    loss = 0
    for ii in range(len(y_true)):
        _,indices= np.unique((mm*y_true[ii]*cache).sum(dim=1).detach().numpy(),return_index=True)
        ## return indices after sorting the array in ascending order
        if mode == 'B':
            big_ind = [indices[0] for p in range(len(indices)-1)] #good one
            small_ind = [indices[p+1] for p in range(len(indices)-1)] #bad one
        if mode == 'W':
            big_ind = [indices[p] for p in range(len(indices)-1)] #good one
            small_ind = [indices[-1] for p in range(len(indices)-1)] #bad one
        if mode == 'S':
            big_ind = [indices[p] for p in range(len(indices)-1)] #good one
            small_ind = [indices[p+1] for p in range(len(indices)-1)] #bad one

        loss += (mm*( torch.matmul(cache[big_ind], y_hat[ii]) - torch.matmul(cache[small_ind], y_hat[ii]) 
    - (torch.matmul(cache[big_ind], y_true[ii]) - torch.matmul(cache[small_ind], y_true[ii])) )).square().mean()
        
    return loss


def Listnet_loss(y_hat,y_true,sol,cache, tau=1.,minimize=False,*wd,**kwd):
    mm = 1 if minimize else -1 
    loss = 0
    for ii in range(len(y_true)):
         loss += -(F.log_softmax((-mm*y_hat[ii]*cache/tau).sum(dim=1),
                dim=0)*F.softmax((-mm*y_true[ii]*cache/tau).sum(dim=1),dim=0)).mean()
    return loss
### function to grow the cache
def growcache(cache, y_hat, m, param):
    '''
    cache is torch array [currentpoolsize,48]
    y_hat is  torch array [batch_size,48]
    '''
    sol =  batch_solve(param,y_hat.detach().numpy(),m.numpy(),relaxation =False) 
    cache_np = cache.detach().numpy()
    cache_np = np.unique(np.append(cache_np,sol,axis=0),axis=0)
    # torch has no unique function, so we have to do this
    return torch.from_numpy(cache_np).float()
class CachingPO(twostage_regression):
    def __init__(self, param, loss_fn, init_cache,growth=0.1, tau=1., lr=1e-1, seed=2, max_epochs=50):
        super().__init__(param, lr, seed, max_epochs)
        # self.save_hyperparameters()
        self.loss_fn = loss_fn
        ### The cache
        init_cache_np = init_cache.detach().numpy()
        init_cache_np = np.unique(init_cache_np,axis=0)
        # torch has no unique function, so we have to do this
        self.cache = torch.from_numpy(init_cache_np).float()
        self.growth = growth
        self.tau = tau
        self.save_hyperparameters("lr","growth","tau")

    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        if (np.random.random(1)[0]< self.growth) or len(self.cache)==0:
            self.cache = growcache(self.cache, y_hat,m, self.param)

        loss = self.loss_fn(y_hat,y,sol,self.cache,tau = self.tau)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class CombinedPO(twostage_regression):
    def __init__(self,loss_fn, init_cache, growpool_fn, growth =0.0, lr=1e-1,tau=0.,alpha=0.5):
        super().__init__(lr)
        # self.save_hyperparameters()
        self.loss_fn = loss_fn
        ### The cache
        init_cache_np = init_cache.detach().numpy()
        init_cache_np = np.unique(init_cache_np,axis=0)
        # torch has no unique function, so we have to do this
        self.cache = torch.from_numpy(init_cache_np).float()
        self.growpool_fn = growpool_fn
        self.growth = growth
        self.tau = tau
        self.alpha = alpha
        self.save_hyperparameters("lr","growth","tau","alpha")
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SemanticPO")
        parser.add_argument("--lr", type=float, default=0.1)
        return parent_parser  
    def training_step(self, batch, batch_idx):
        x,y,sol,m = batch
        y_hat =  self(x).squeeze()
        if (np.random.random(1)[0]< self.growth) or len(self.cache)==0:
            self.cache = self.growpool_fn(self.cache, y_hat,m)

        criterion = nn.MSELoss(reduction='mean')
        loss = self.alpha*self.loss_fn(y_hat,y,sol,self.cache,tau =self.tau) + (1- self.alpha)*criterion(y_hat,y)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class CoraDataset():
    def __init__(self, x,y, M, param={'p':0.25, 'q':0.25},  relaxation=False, sols=None, verbose=False):

        if sols is not None:
            self.sols = sols
        else:
            y_iter = range(len(y))
            it = tqdm(y_iter) if verbose else y_iter
            self.sols =  batch_solve(param,y,M,relaxation =False)  
            self.sols = torch.from_numpy(self.sols).float()
        
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.m = torch.from_numpy(M).float()
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.sols[index], self.m[index]