from PO_model_matching import *
import shutil
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import random
torch.use_deterministic_algorithms(True)

exp = "Matching1"
param_dict = {"Matching1":{'p':0.25, 'q':0.25},"Matching2":{'p':0.5, 'q':0.5} }
param = param_dict[exp]
def seed_all(seed):
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
outputfile = "Rslt/NCE_{}_rslt.csv".format(exp)
regretfile= "Rslt/NCE_{}_Regret.csv".format(exp)
ckpt_dir =  "ckpt_dir/NCE_{}/".format(exp)
log_dir = "lightning_logs/NCE_{}/".format(exp)
shutil.rmtree(log_dir,ignore_errors=True)
###################################### Hyperparams #########################################
lr = 1e-2
batchsize  = 32
max_epochs = 30
growth = 0.1
######################################  Data Reading #########################################
x, y,m = get_cora()
x_train, x_test = x[:22], x[22:]
y_train, y_test = y[:22], y[22:]
m_train, m_test = m[:22], m[22:]
train_df = CoraDataset( x_train,y_train,m_train,param=param)
valid_df = CoraDataset( x_test,y_test,m_test, param=param)

train_dl = DataLoader(train_df, batch_size= batchsize)
valid_dl = DataLoader(valid_df, batch_size=5)
print(y_train.shape, y_test.shape)

cache_np = batch_solve(param,y_train,m,relaxation =False) 
cache_np = np.unique(cache_np, axis=0)
cache =  torch.from_numpy(cache_np).float()
for seed in range(10):
    seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    train_dl = DataLoader(train_df, batch_size= batchsize,generator=g, num_workers=8)
    valid_dl = DataLoader(valid_df, batch_size=5)


    shutil.rmtree(ckpt_dir,ignore_errors=True)
    checkpoint_callback = ModelCheckpoint(
                    monitor="val_regret",
                    dirpath= ckpt_dir,
                    filename="model-{epoch:02d}-{val_loss:.2f}",
                    save_top_k=1,save_last = True,
                    mode="min")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)
                

    trainer = pl.Trainer(max_epochs= max_epochs, callbacks=[checkpoint_callback], min_epochs=5, logger=tb_logger)

    model = CachingPO( param=param,loss_fn= NCE_hatc_c,init_cache=cache, growth=growth,
     lr=lr, seed=seed, max_epochs=max_epochs )
    trainer.fit(model, train_dl, valid_dl)
    best_model_path = checkpoint_callback.best_model_path
    model = CachingPO.load_from_checkpoint(best_model_path,
    param=param,loss_fn= NCE_hatc_c,init_cache=cache, growth=growth,
     lr=lr, seed=seed, max_epochs=max_epochs)
    y_pred = model(torch.from_numpy(x_test).float()).squeeze()
    sol_test = batch_solve(param,y_test,m_test)
    regret_list = regret_aslist( y_pred, torch.from_numpy(y_test).float(),
     torch.from_numpy(sol_test).float(), torch.from_numpy(m_test).float(), param )
    df = pd.DataFrame({"regret":regret_list})
    df.index.name='instance'
    df ['model'] = 'NCE_hatc_c'
    df['seed'] = seed
    df ['batchsize'] = batchsize
    df['growth'] = growth
 
    df['lr'] = lr
    with open(regretfile, 'a') as f:
        df.to_csv(f, header=f.tell()==0)

    ##### Summary
    result = trainer.test(model, dataloaders= valid_dl)
    df = pd.DataFrame(result)
    df ['model'] = 'NCE_hatc_c'
    df['seed'] = seed
    df ['batchsize'] = batchsize
    df['growth'] = growth


    df['lr'] = lr
    with open(outputfile, 'a') as f:
        df.to_csv(f, header=f.tell()==0)
###############################  Save  Learning Curve Data ########
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
parent_dir=   log_dir+"default/"
version_dirs = [os.path.join(parent_dir,v) for v in os.listdir(parent_dir)]

walltimes = []
steps = []
regrets= []
mses = []
aucs = []
for logs in version_dirs:
    event_accumulator = EventAccumulator(logs)
    event_accumulator.Reload()

    events = event_accumulator.Scalars("val_regret_epoch")
    walltimes.extend( [x.wall_time for x in events])
    steps.extend([x.step for x in events])
    regrets.extend([x.value for x in events])
    events = event_accumulator.Scalars("val_mse_epoch")
    mses.extend([x.value for x in events])
    events = event_accumulator.Scalars("val_auc_epoch")
    aucs.extend([x.value for x in events])
df = pd.DataFrame({"step": steps,'wall_time':walltimes,  "val_regret": regrets,
 "val_mse": mses ,"val_auc":aucs})
df['model'] = 'NCE_hatc_c'
df.to_csv("LearningCurve/NCE_{}_growth{}_lr{}.csv".format(exp,growth,lr))