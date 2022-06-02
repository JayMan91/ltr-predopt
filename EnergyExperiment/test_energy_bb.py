from PO_model_energy import *
from sklearn.preprocessing import StandardScaler
from get_energy import get_energy
import shutil
import random
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint 
from pytorch_lightning import loggers as pl_loggers
import pandas as pd
torch.use_deterministic_algorithms(True)
load = 2
param = data_reading("EnergyCost/load{}/day01.txt".format(load))

def seed_all(seed):
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

outputfile = "Rslt/Blackbox_rslt_load{}.csv".format(load)
regretfile= "Rslt/Blackbox_Regret_load{}.csv".format(load)
ckpt_dir =  "ckpt_dir/Blackbox_load{}/".format(load)
log_dir = "lightning_logs/Blackbox_load{}/".format(load)

shutil.rmtree(log_dir,ignore_errors=True)
###################################### Hyperparams #########################################
lr = 0.1
mu = 1e-1
batchsize  = 128
max_epochs = 20

######################################  Data Reading #########################################
x_train, y_train, x_test, y_test = get_energy(fname= 'prices2013.dat')
x_train = x_train[:,1:]
x_test = x_test[:,1:]
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(-1,48,x_train.shape[1])
y_train = y_train.reshape(-1,48)
sol_train =  batch_solve(param,y_train,relax=False)
x_test = x_test.reshape(-1,48,x_test.shape[1])
y_test = y_test.reshape(-1,48)
sol_test =  batch_solve(param,y_test,relax=False)
x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train,y_test), axis=0)
sol = np.concatenate((sol_train,sol_test), axis=0)
n_samples =  len(x)
for seed in range(10):
    seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    randomize = np.arange(n_samples)
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize] 
    sol = sol[randomize]
    x_train, y_train, sol_train = x[:552], y[:552], sol[:552]
    x_valid, y_valid, sol_valid = x[552:652], y[552:652], sol[552:652]
    x_test, y_test, sol_test = x[652:], y[652:], sol[652:]
    print(x_train.shape, x_valid.shape, x_test.shape)
    print(sol_train.shape,sol_valid.shape, sol_test.shape)


    train_df = EnergyDatasetWrapper( x_train,y_train,param, sol=sol_train, relax=False)
    cache_np = np.unique(sol_train, axis=0)
    cache =  torch.from_numpy(cache_np).float()

    train_dl = DataLoader(train_df, batch_size= batchsize, generator=g)

    valid_df = EnergyDatasetWrapper( x_valid,y_valid,param, sol=sol_valid,  relax=False)
    valid_dl = DataLoader(valid_df, batch_size= 50)

    test_df = EnergyDatasetWrapper( x_test,y_test,param, sol=sol_test, relax=False)
    test_dl = DataLoader(test_df, batch_size= 100)
    shutil.rmtree(ckpt_dir,ignore_errors=True)
    checkpoint_callback = ModelCheckpoint(
            monitor="val_regret",
            dirpath= ckpt_dir,
            filename="model-{epoch:02d}-{val_loss:.2f}",
            save_top_k=2,save_last = True,
            mode="min",
        )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)

    trainer = pl.Trainer(max_epochs= max_epochs,callbacks=[checkpoint_callback],  min_epochs=1, logger=tb_logger)
    model = Blackbox(param=param,net=nn.Linear(8,1),lr=lr,mu=mu, seed=seed, max_epochs= max_epochs)


    trainer.fit(model, train_dl, valid_dl)
    best_model_path = checkpoint_callback.best_model_path
    model = Blackbox.load_from_checkpoint(best_model_path,
    param=param,net=nn.Linear(8,1),lr=lr,mu=mu, seed=seed, max_epochs= max_epochs)

    y_pred = model(torch.from_numpy(x_test).float()).squeeze()
    
    regret_list = regret_aslist(y_pred,torch.from_numpy(y_test).float(), 
    torch.from_numpy(sol_test).float(), param, minimize=True)

    df = pd.DataFrame({"regret":regret_list})
    df.index.name='instance'
    df ['model'] = 'Blackbox'
    df['seed'] = seed
    df ['batchsize'] = batchsize
    df['lr'] = lr
    df['mu'] = mu
    with open(regretfile, 'a') as f:
        df.to_csv(f, header=f.tell()==0)
    ##### Summary
    result = trainer.test(model, dataloaders=test_dl)
    df = pd.DataFrame(result)
    df ['model'] = 'Blackbox'
    df['seed'] = seed
    df ['batchsize'] = batchsize
    df['lr'] = lr
    df['mu'] = mu
    with open(outputfile, 'a') as f:
        df.to_csv(f, header=f.tell()==0)
###############################  Save  Learning Curve Data ###############################
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
parent_dir=   log_dir+"default/"
version_dirs = [os.path.join(parent_dir,v) for v in os.listdir(parent_dir)]

walltimes = []
steps = []
regrets= []
mses = []
for logs in version_dirs:
    event_accumulator = EventAccumulator(logs)
    event_accumulator.Reload()

    events = event_accumulator.Scalars("val_regret_epoch")
    walltimes.extend( [x.wall_time for x in events])
    steps.extend([x.step for x in events])
    regrets.extend([x.value for x in events])
    events = event_accumulator.Scalars("val_mse_epoch")
    mses.extend([x.value for x in events])

df = pd.DataFrame({"step": steps,'wall_time':walltimes,  "val_regret": regrets,
 "val_mse": mses })
df['model'] ='Blackbox'
df.to_csv("LearningCurve/Blackbox_load{}.csv".format(load))