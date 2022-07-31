import os
import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import networkx as nx
import gurobipy as gp

##################################### Graph Structure ###################################################
V = range(25)
E = []

for i in V:
    if (i+1)%5 !=0:
        E.append((i,i+1))
    if i+5<25:
            E.append((i,i+5))

G = nx.DiGraph()
G.add_nodes_from(V)
G.add_edges_from(E)

###################################### Gurobi Shortest path Solver #########################################
class shortestpath_solver:
    def __init__(self,G= G):
        self.G = G
    
    def shortest_pathsolution(self, y):
        '''
        y the vector of  edge weight
        '''
        A = nx.incidence_matrix(G,oriented=True).todense()
        b =  np.zeros(len(A))
        b[0] = -1
        b[-1] =1
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        x = model.addMVar(shape=A.shape[1], vtype=gp.GRB.BINARY, name="x")
        model.setObjective(y @x, gp.GRB.MINIMIZE)
        model.addConstr(A @ x == b, name="eq")
        model.optimize()
        if model.status==2:
            return x.x
    def solution_fromtorch(self,y_torch):
        return torch.from_numpy(self.shortest_pathsolution( y_torch.detach().numpy())).float()
spsolver =  shortestpath_solver()
###################################### Wrapper #########################################
class datawrapper():
    def __init__(self, x,y, sol=None, solver= spsolver ):
        self.x = x
        self.y = y
        if sol is None:
            sol = []
            for i in range(len(y)):
                sol.append(   solver.shortest_pathsolution(y[i])   )            
            sol = np.array(sol).astype(np.float32)
        self.sol = sol

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index],self.sol[index]


def batch_solve(solver, y,relaxation =False):
    sol = []
    for i in range(len(y)):
        sol.append(   solver.solution_fromtorch(y[i]).reshape(1,-1)   )
    return torch.cat(sol,0)


def regret_fn(solver, y_hat,y, minimize= True):  
    '''
    computes regret given predicted y_hat and true y
    '''
    regret_list = []
    for ii in range(len(y)):
        regret_list.append( SPOLoss(solver, minimize)(y_hat[ii],y[ii]) )
    return torch.mean( torch.tensor(regret_list ))

def regret_aslist(solver, y_hat,y, minimize= True):  
    '''
    computes regret of more than one cost vectors
    ''' 
    regret_list = []
    for ii in range(len(y)):
        regret_list.append( SPOLoss(solver, minimize)(y_hat[ii],y[ii]).item() )
    return np.array(regret_list)

class twostage_regression(pl.LightningModule):
    def __init__(self,net,exact_solver = spsolver, lr=1e-1, l1_weight=0.1,max_epochs=30, seed=20):
        """
        A class to implement two stage mse based model and with test and validation module
        Args:
            net: the neural network model
            exact_solver: the solver which returns a shortest path solution given the edge cost
            lr: learning rate
            l1_weight: the lasso regularization weight
            max_epoch: maximum number of epcohs
            seed: seed for reproducibility 
        """
        super().__init__()
        pl.seed_everything(seed)
        self.net =  net
        self.lr = lr
        self.l1_weight = l1_weight
        self.exact_solver = exact_solver
        self.max_epochs= max_epochs
        self.save_hyperparameters("lr",'l1_weight')
    def forward(self,x):
        return self.net(x) 
    def training_step(self, batch, batch_idx):
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(y_hat,y)
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        training_loss =  loss  + l1penalty * self.l1_weight
        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",loss,  on_step=True, on_epoch=True, )
        return training_loss 
    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss(reduction='mean')
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        mseloss = criterion(y_hat, y)
        regret_loss =  regret_fn(self.exact_solver, y_hat,y) 
        pointwise_loss = pointwise_crossproduct_loss(y_hat,y)

        self.log("val_mse", mseloss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("val_regret", regret_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("val_pointwise", pointwise_loss, prog_bar=True, on_step=True, on_epoch=True, )

        return {"val_mse":mseloss, "val_regret":regret_loss}
    def validation_epoch_end(self, outputs):
        avg_regret = torch.stack([x["val_regret"] for x in outputs]).mean()
        avg_mse = torch.stack([x["val_mse"] for x in outputs]).mean()
        
        self.log("ptl/val_regret", avg_regret)
        self.log("ptl/val_mse", avg_mse)
        # self.log("ptl/val_accuracy", avg_acc)
        
    def test_step(self, batch, batch_idx):
        # num = np.random.random(11)
        # print("test number",num)
        # self.log("length", num, prog_bar=True, on_step=True, on_epoch=True,)
        # # return {"length":len(batch)}
        return self.validation_step(batch, batch_idx)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        num_batches = len(self.train_dataloader()) // self.trainer.accumulate_grad_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.lr, epochs=self.max_epochs,
        steps_per_epoch = num_batches)
        return [optimizer], [scheduler]

###################################### SPO and Blackbox #########################################

def SPOLoss(solver, minimize=True):
    mm = 1 if minimize else -1
    class SPOLoss_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y_pred, y_true, sol_true):
       
            sol_hat = solver.solution_fromtorch(y_pred)
            sol_spo = solver.solution_fromtorch(2* y_pred - y_true)
            # sol_true = solver.solution_fromtorch(y_true)
            ctx.save_for_backward(sol_spo,  sol_true, sol_hat)
            return   mm*(  sol_hat - sol_true).dot(y_true)/( sol_true.dot(y_true) ) # changed to per cent rgeret

        @staticmethod
        def backward(ctx, grad_output):
            sol_spo,  sol_true, sol_hat = ctx.saved_tensors
            return mm*(sol_true - sol_spo), None, None
            
    return SPOLoss_cls.apply
def BlackboxLoss(solver,mu=0.1, minimize=True):
    mm = 1 if minimize else -1
    class BlackboxLoss_cls(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y_pred, y_true, sol_true):
            sol_hat = solver.solution_fromtorch(y_pred)
            sol_perturbed = solver.solution_fromtorch(y_pred + mu* y_true)
            # sol_true = solver.solution_fromtorch(y_true)
            ctx.save_for_backward(sol_perturbed,  sol_true, sol_hat)
            return   mm*(  sol_hat - sol_true).dot(y_true)/( sol_true.dot(y_true) ) # changed to per cent rgeret

        @staticmethod
        def backward(ctx, grad_output):
            sol_perturbed,  sol_true, sol_hat = ctx.saved_tensors
            return -mm*(sol_hat - sol_perturbed)/mu, None, None
            
    return BlackboxLoss_cls.apply



class SPO(twostage_regression):
    def __init__(self,net,exact_solver = spsolver,lr=1e-1, l1_weight=0.1,max_epochs=30, seed=20):
        """
        Implementaion of SPO+ loss subclass of twostage model
 
        """
        super().__init__(net,exact_solver, lr, l1_weight,max_epochs, seed)
    def training_step(self, batch, batch_idx):
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        loss = 0
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        for ii in range(len(y)):
            loss += SPOLoss(self.exact_solver)(y_hat[ii],y[ii], sol[ii])
        training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",loss/len(y),  on_step=True, on_epoch=True, )
        return training_loss  

class Blackbox(twostage_regression):
    """
    Implemenation of Blackbox differentiation gradient
    """
    def __init__(self,net,exact_solver = spsolver,lr=1e-1,mu =0.1, l1_weight=0.1,max_epochs=30, seed=20):
        super().__init__(net,exact_solver , lr, l1_weight,max_epochs, seed)
        self.mu = mu
        self.save_hyperparameters("lr","mu")
    def training_step(self, batch, batch_idx):
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        loss = 0
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])

        for ii in range(len(y)):
            loss += BlackboxLoss(self.exact_solver,self.mu)(y_hat[ii],y[ii], sol[ii])
        training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",loss/len(y),  on_step=True, on_epoch=True, )
        return training_loss   


###################################### Ranking Loss  #########################################
###################################### Ranking Loss  Functions  #########################################

def pointwise_mse_loss(y_hat,y_true):
    c_hat = y_hat.unsqueeze(-1)
    c_true = y_true.unsqueeze(-1)
    c_diff = c_hat - c_true
    loss = ( c_diff.square().sum())/len(c_diff)
    return loss   

def pointwise_crossproduct_loss(y_hat,y_true):
    c_hat = y_hat.unsqueeze(-1)
    c_true = y_true.unsqueeze(-1)
    c_diff = c_hat - c_true
    loss = (torch.bmm(c_diff, c_diff.transpose(2,1)).sum() )/len(c_diff)
    return loss   

def pointwise_custom_loss(y_hat,y_true, *wd,**kwd):
    loss =  pointwise_mse_loss(y_hat,y_true) + pointwise_crossproduct_loss(y_hat,y_true)
    return loss 



def pointwise_loss(y_hat,y_true,sol_true, cache,*wd,**kwd):
    '''
    sol, y and y_hat are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    f(y_hat,s) is regresson on f(y,s)
    '''
    loss = (torch.matmul(y_hat,cache.transpose(0,1))- torch.matmul(y_true,cache.transpose(0,1))).square().mean()

    return loss



def pairwise_loss(y_hat,y_true,sol_true, cache,tau=0, minimize=True,mode= 'B'):
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

def pairwise_diffloss(y_hat,y_true,sol_true, cache,tau=0, minimize=True,mode= 'B'):
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

def Listnet_loss(y_hat,y_true,sol_true, cache,tau=1,minimize=True,*wd,**kwd):
    mm = 1 if minimize else -1 
    loss = 0
    for ii in range(len(y_true)):
         loss += -(F.log_softmax((-mm*y_hat[ii]*cache/tau).sum(dim=1),
                dim=0)*F.softmax((-mm*y_true[ii]*cache/tau).sum(dim=1),dim=0)).mean()
    return loss
def Listnet_KLloss(y_hat,y_true,sol_true,cache,tau=1,minimize=True,*wd,**kwd):
    mm = 1 if minimize else -1 
    loss = 0
    for ii in range(len(y_true)):
         loss += ( F.log_softmax((-mm*y_true[ii]*cache/tau).sum(dim=1),dim=0) -
         F.log_softmax((-mm*y_hat[ii]*cache).sum(dim=1),
                dim=0)*F.softmax((-mm*y_true[ii]*cache/tau).sum(dim=1),dim=0)).mean()
    return loss

def MAP(y_tilde,sol_true, cache,minimize=True):
    '''
    sol, and y are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    '''
    mm = 1 if minimize else -1 
    loss = 0
    # print("shape check", sol.shape, y.shape,y_hat.shape, cache.shape)
    for ii in range(len(y_tilde)):
        loss += torch.max(((sol_true[ii] - cache )*(mm*y_tilde[ii]  )).sum(dim=1))
    return loss
def MAP_c(y_hat,y_true,sol_true, cache,minimize=True,*wd,**kwd):
    y = y_hat 
    return MAP(sol_true,y,cache,minimize)
def MAP_hatc_c(y_hat,y_true,sol_true, cache,minimize=True,*wd,**kwd):
    y = y_hat - y_true
    return MAP(sol_true,y,cache,minimize)

def NCE(y_tilde,sol_true, cache,minimize=True):
    '''
    sol, and y are torch array [batch_size,48]
    cache is torch array [currentpoolsize,48]
    '''
    mm = 1 if minimize else -1 
    loss = 0
    # print("shape check", sol.shape, y.shape,y_hat.shape, cache.shape)
    for ii in range(len(y_tilde)):
        loss += torch.mean(((sol_true[ii] - cache )*(mm*y_tilde[ii]  )).sum(dim=1))
    return loss
def NCE_c(y_hat,y_true,sol_true,cache,minimize=True,*wd,**kwd):
    y = y_hat 
    return NCE(sol_true,y,cache,minimize)
def NCE_hatc_c(y_hat,y_true,sol_true,cache,minimize=True,*wd,**kwd):
    y = y_hat - y_true
    return NCE(sol_true,y,cache,minimize)

def growcache(solver, cache, y_hat):
    '''
    cache is torch array [currentpoolsize,48]
    y_hat is  torch array [batch_size,48]
    '''
    sol = batch_solve(solver, y_hat,relaxation =False).detach().numpy()
    cache_np = cache.detach().numpy()
    cache_np = np.unique(np.append(cache_np,sol,axis=0),axis=0)
    # torch has no unique function, so we need to do this
    return torch.from_numpy(cache_np).float()


class CachingPO(twostage_regression):
    def __init__(self,loss_fn,init_cache, net,exact_solver = spsolver,growth=0.1,tau=0.,lr=1e-1,
        l1_weight=0.1,max_epochs=30, seed=20):
        """
        A class to implement loss functions using soluton cache
        Args:
            loss_fn: the loss function (NCE, MAP or the rank-based ones)
            init_cache: initial solution cache
            growth: p_solve
            tau: the margin parameter for pairwise ranking / temperatrure for listwise ranking
            net: the neural network model
            exact_solver: the solver which returns a shortest path solution given the edge cost
            lr: learning rate
            l1_weight: the lasso regularization weight
            max_epoch: maximum number of epcohs
            seed: seed for reproducibility 

        """
        super().__init__(net,exact_solver, lr, l1_weight,max_epochs, seed)
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
        x,y, sol = batch
        y_hat =  self(x).squeeze()
        if (np.random.random(1)[0]<= self.growth) or len(self.cache)==0:
            self.cache = growcache(self.exact_solver, self.cache, y_hat)

        loss = self.loss_fn(y_hat,y,sol, self.cache, self.tau)
        l1penalty = sum([(param.abs()).sum() for param in self.net.parameters()])
        training_loss=  loss/len(y)  + l1penalty * self.l1_weight
        self.log("train_totalloss",training_loss, prog_bar=True, on_step=True, on_epoch=True, )
        self.log("train_l1penalty",l1penalty * self.l1_weight,  on_step=True, on_epoch=True, )
        self.log("train_loss",loss/len(y),  on_step=True, on_epoch=True, )
        return training_loss  

