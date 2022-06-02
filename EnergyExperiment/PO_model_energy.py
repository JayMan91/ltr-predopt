import os
import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import gurobipy as gp
from ICON_solving import *
from get_energy import get_energy

def MakeLpMat(nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,**h):
    # nbMachines: number of machine
    # nbTasks: number of task
    # nb resources: number of resources
    # MC[m][r] resource capacity of machine m for resource r 
    # U[f][r] resource use of task f for resource r
    # D[f] duration of tasks f
    # E[f] earliest start of task f
    # L[f] latest end of task f
    # P[f] power use of tasks f
    # idle[m] idle cost of server m
    # up[m] startup cost of server m
    # down[m] shut-down cost of server m
    # q time resolution
    # timelimit in seconds
    """
    G1: rows: n_machine * Time; cols: n_task*n_machine* Time
        first T row for machine1, next T: (2T) for machine 2 and so on
        first n_task column for task 1 of machine 1 in time slot 0 then for task 1 machine 2 and so on
    x: decisiion variable-vector of n_task*n_machine* Time. x[  f*(n_task*n_machine* Time)+m*(n_machine* Time)+Time ]=1 if task f starts at time t on machine m.
    A1: To ensure each task is scheduled only once.
    A2: To respect early start time
    A3: To respect late start time
    F: rows:Time , cols: n_task*n_machine* Time, bookkeping for power power use for each time unit
    Code is written assuming nb resources=1
    """
    Machines = range(nbMachines)
    Tasks = range(nbTasks)
    Resources = range(nbResources)
    N = 1440//q

    ### G and h
    G1 = torch.zeros((nbMachines*N,nbTasks*nbMachines*N)).float()
    h1 = torch.zeros(nbMachines*N).float()
    F = torch.zeros((N,nbTasks*nbMachines*N)).float()
    for m in Machines:
        for t in range(N):
            ## in all of our problem, we have only one resource
            h1[m*N+t] = MC[m][0]
            for f in Tasks:
                c_index = (f*nbMachines+m)*N 
                G1[t + m*N, (c_index+max(0,t-D[f]+1)):(c_index+(t+1))] = U[f][0]
                F [t,(c_index+max(0,t-D[f]+1)):(c_index+(t+1))  ] = P[f]

    G2 = torch.eye((nbTasks*nbMachines*N))
    G3 = -1*torch.eye((nbTasks*nbMachines*N))
    h2 = torch.ones(nbTasks*nbMachines*N)
    h3 = torch.zeros(nbTasks*nbMachines*N)

    G = G1 # torch.cat((G1,G2,G3)) 
    h = h1 # torch.cat((h1,h2,h3))
    ### A and b
    A1 = torch.zeros((nbTasks, nbTasks*nbMachines*N)).float()
    A2 = torch.zeros((nbTasks, nbTasks*nbMachines*N)).float()
    A3 = torch.zeros((nbTasks, nbTasks*nbMachines*N)).float()

    for f in Tasks:
        A1 [f,(f*N*nbMachines):((f+1)*N*nbMachines) ] = 1
        for m in Machines:
            start_index = f*N*nbMachines + m*N # Time 0 for task f machine m
            ## early start time
            A2 [f,start_index:( start_index + E[f]) ] = 1
            ## latest end time
            A3 [f,(start_index+L[f]-D[f]+1):(start_index+N) ] = 1
    b = torch.cat((torch.ones(nbTasks),torch.zeros(2*nbTasks)))
    A = torch.cat((A1,A2,A3))    
    return A,b,G,h,torch.transpose(F, 0, 1)

def IconMatrixsolver(A,b,G,h,F,y):
    '''
    A,b,G,h define the problem
    y: the price of each hour
    Multiply y with F to reach the granularity of x
    x is the solution vector for each hour for each machine for each task 
    '''
    n = A.shape[1]
    m = gp.Model("matrix1")
    x = m.addMVar(shape=n, vtype=GRB.BINARY, name="x")

    m.addConstr(A @ x == b, name="eq")
    m.addConstr(G @ x <= h, name="ineq")
    c  = np.matmul(F,y).squeeze()
    m.setObjective(c @ x, GRB.MINIMIZE)
    m.optimize()
    if m.status==2:
        return x.X


def batch_solve(param,y,relax=False):
    '''
    wrapper around te solver to return solution of a vector of cost coefficients
    '''
    clf =  SolveICON(relax=relax,**param)
    clf.make_model()
    sol = []
    for i in range(len(y)):
        sol.append( clf.solve_model(y[i]))
    return np.array(sol)

def regret_fn(y_hat,y, sol_true,param, minimize=True):
    '''
    computes average regret given a predicted cost vector and the true solution vector and the true cost vector
    y_hat,y, sol_true are torch tensors
    '''
    mm = 1 if minimize else -1    
    sol_hat = torch.from_numpy(batch_solve(param,y_hat.detach().numpy()))
    return  ((mm*(sol_hat - sol_true)*y).sum(1)/(sol_true*y).sum(1)).mean()

def regret_aslist(y_hat,y, sol_true,param, minimize=True): 
    '''
    computes regret of more than one cost vectors
    ''' 
    mm = 1 if minimize else -1    
    sol_hat = torch.from_numpy(batch_solve(param,y_hat.detach().numpy()))
    return  ((mm*(sol_hat - sol_true)*y).sum(1)/(sol_true*y).sum(1))
class EnergyDatasetWrapper():
    def __init__(self, X,y,param,sol=None, relax=False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        if sol is None:
            sol = batch_solve(param, y, relax)

        self.sol = np.array(sol).astype(np.float32)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx],self.y[idx],self.sol[idx]



class twostage_regression(pl.LightningModule):
    def __init__(self,param,net, lr=1e-1, max_epochs=30, seed=20):
        """
        A class to implement two stage mse based model and with test and validation module
        Args:
            net: the neural network model
            param: the parameter of the scheduling problem
            lr: learning rate
            max_epochs: maximum number of epcohs
            seed: seed for reproducibility 
        """
        super().__init__()
        pl.seed_everything(seed)
        self.net = net
        self.param = param
        self.lr = lr
        self.max_epochs= max_epochs
        self.save_hyperparameters("lr")

    def forward(self,x):
        return self.net(x) 
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(y_hat,y)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction='mean')
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        val_loss= regret_fn(y_hat,y, sol,self.param)
        mseloss = criterion(y_hat, y)
        self.log("val_regret", val_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("val_mse", mseloss, prog_bar=True, on_step=True, on_epoch=True, )
       
        return  {"val_regret": val_loss, "val_mse": mseloss}
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
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        # # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=1)
        return [optimizer], [scheduler]

def spograd(y,sol,param,minimize=True):
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
            sol_spo =   torch.from_numpy(batch_solve(param,y_spo.detach().numpy()))
            return (sol- sol_spo)*mm
    return spograd_cls.apply

def bbgrad(y,sol,param,mu,minimize=True):
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
            sol_hat = torch.from_numpy(batch_solve(param,y_hat.detach().numpy()))
            sol_perturbed = torch.from_numpy(batch_solve(param,y_perturbed.detach().numpy()))
            return -mm*(sol_hat - sol_perturbed)/mu 
    return bbgrad_cls.apply
class SPO(twostage_regression):
    def __init__(self,param,net, lr=1e-1, max_epochs=30, seed=20):
        super().__init__(param,net, lr, max_epochs, seed)
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        loss = spograd(y,sol,self.param)(y_hat)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss
class Blackbox(twostage_regression):
    def __init__(self,param,net,mu, lr=1e-1, max_epochs=30, seed=20):
        super().__init__(param,net, lr, max_epochs, seed)
        self.mu = mu
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        loss = bbgrad(y,sol,self.param,self.mu)(y_hat)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss




###################################### Ranking Loss Functions #########################################

def pointwise_loss(y_hat,y_true,sol,cache,*wd,**kwd):
    '''
    sol, y and y_hat are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    '''
    loss = (torch.matmul(y_hat,cache.transpose(0,1))- torch.matmul(y_true,cache.transpose(0,1))).square().mean()
    return loss
def pairwise_diffloss(y_hat,y_true,sol,cache, minimize=True,mode= 'B',*wd,**kwd):
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
def pairwise_loss(y_hat,y_true,sol,cache,tau=0,minimize=True,mode= 'B'):
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


def Listnet_loss(y_hat,y_true,sol,cache,tau=1, minimize=True,*wd,**kwd):
    mm = 1 if minimize else -1 
    loss = 0
    for ii in range(len(y_true)):
         loss += -(F.log_softmax((-mm*y_hat[ii]*cache/tau).sum(dim=1),
                dim=0)*F.softmax((-mm*y_true[ii]*cache/tau).sum(dim=1),dim=0)).mean()
    return loss

def Listnet_KLloss(y_hat,y_true,sol,cache,tau=1,minimize=True,*wd,**kwd):
    mm = 1 if minimize else -1 
    loss = 0
    for ii in range(len(y_true)):
         loss += ( F.log_softmax((-mm*y_true[ii]*cache).sum(dim=1),dim=0) -
         F.log_softmax((-mm*y_hat[ii]*cache).sum(dim=1),
                dim=0)*F.softmax((-mm*y_true[ii]*cache).sum(dim=1),dim=0)).mean()
    return loss


def MAP(sol,y,cache,minimize=True):
    '''
    sol, y and y_hat are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    '''
    mm = 1 if minimize else -1 
    loss = 0
    # print("shape check", sol.shape, y.shape,y_hat.shape, cache.shape)
    for ii in range(len(y)):
        loss += torch.max(((sol[ii] - cache )*(y[ii]*mm  )).sum(dim=1))
    return loss
def MAP_c(y_hat,y_true,sol,cache,minimize=True,*wd,**kwd):
    y = y_hat 
    return MAP(sol,y,cache,minimize)
def MAP_hatc_c(y_hat,y_true,sol,cache,minimize=True,*wd,**kwd):
    y = y_hat - y_true
    return MAP(sol,y,cache,minimize)

def NCE(sol,y,cache,minimize=True):
    '''
    sol, y and y_hat are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    '''
    mm = 1 if minimize else -1 
    loss = 0
    # print("shape check", sol.shape, y.shape,y_hat.shape, cache.shape)
    for ii in range(len(y)):
        loss += torch.mean(((sol[ii] - cache )*(y[ii]*mm  )).sum(dim=1))
    return loss
def NCE_c(y_hat,y_true,sol,cache,minimize=True,*wd,**kwd):
    y = y_hat 
    return NCE(sol,y,cache,minimize)
def NCE_hatc_c(y_hat,y_true,sol,cache,minimize=True,*wd,**kwd):
    y = y_hat - y_true
    return NCE(sol,y,cache,minimize)

### function to grow the cache
def growcache(cache, y_hat,param):
    '''
    cache is torch array [currentpoolsize,48]
    y_hat is  torch array [batch_size,48]
    '''
    sol = batch_solve(param,y_hat)
    cache_np = cache.detach().numpy()
    cache_np = np.unique(np.append(cache_np,sol,axis=0),axis=0)
    # torch has no unique function, so we have to do this
    return torch.from_numpy(cache_np).float()


class CachingPO(twostage_regression):
    def __init__(self,loss_fn,param,net,init_cache, growth =0.1, lr=1e-1,tau=0.,
        max_epochs=30, seed=20):
        '''
        tau: the margin parameter for pairwise ranking / temperatrure for listwise ranking
        '''
        super().__init__(param,net, lr, max_epochs, seed)
        self.loss_fn = loss_fn
        # self.save_hyperparameters()
  
        ### The cache
        init_cache_np = init_cache.detach().numpy()
        init_cache_np = np.unique(init_cache_np,axis=0)
        # torch has no unique function, so we have to do this
        self.cache = torch.from_numpy(init_cache_np).float()
        self.growth = growth
        self.tau = tau
        self.save_hyperparameters("lr","growth","tau")
    
 
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        if (np.random.random(1)[0]< self.growth) or len(self.cache)==0:
            self.cache = growcache(self.cache, y_hat, self.param)

        loss = self.loss_fn(y_hat,y,sol,self.cache,tau=self.tau)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss

class CombinedPO(CachingPO):
    def __init__(self,alpha, loss_fn,param,net,init_cache, growth =0.1, lr=1e-1,tau=0.,
        max_epochs=30, seed=20):
        '''
        tau: the margin parameter for pairwise ranking / temperatrure for listwise ranking
        '''
        super().__init__(loss_fn,param,net,init_cache, growth , lr,tau, max_epochs, seed)
        self.alpha = alpha
        self.save_hyperparameters("lr","growth","tau","alpha")
    
 
    def training_step(self, batch, batch_idx):
        x,y,sol = batch
        y_hat =  self(x).squeeze()
        if (np.random.random(1)[0]< self.growth) or len(self.cache)==0:
            self.cache = growcache(self.cache, y_hat, self.param)
        criterion = nn.MSELoss(reduction='mean')
        loss = self.alpha* self.loss_fn(y_hat,y,sol,self.cache,tau=self.tau) + (1 - self.alpha)*criterion(y_hat,y)
        self.log("train_loss",loss, prog_bar=True, on_step=True, on_epoch=True, )
        return loss