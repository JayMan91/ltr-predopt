from PO_modelsSP import *
import pandas as pd
from torch.utils.data import DataLoader
import shutil
from pytorch_lightning.callbacks import ModelCheckpoint 
import random
from pytorch_lightning import loggers as pl_loggers
torch.use_deterministic_algorithms(True)
def seed_all(seed):
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

outputfile = "Rslt/SPO_rslt.csv"
regretfile= "Rslt/SPO_Regret.csv"
ckpt_dir =  "ckpt_dir/SPO/"
log_dir = "lightning_logs/SPO/"
shutil.rmtree(log_dir,ignore_errors=True)

############### Configuration
N, noise, deg = 1000,0.5,6
###################################### Hyperparams #########################################
lr = 0.9
l1_weight = 1e-5
batchsize  = 128
max_epochs = 40
######################################  Data Reading #########################################

Train_dfx= pd.read_csv("SyntheticData/TraindataX_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None)
Train_dfy= pd.read_csv("SyntheticData/Traindatay_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None,)
x_train =  Train_dfx.T.values.astype(np.float32)
y_train = Train_dfy.T.values.astype(np.float32)

Validation_dfx= pd.read_csv("SyntheticData/ValidationdataX_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None)
Validation_dfy= pd.read_csv("SyntheticData/Validationdatay_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None,)
x_valid =  Validation_dfx.T.values.astype(np.float32)
y_valid = Validation_dfy.T.values.astype(np.float32)

Test_dfx= pd.read_csv("SyntheticData/TestdataX_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None)
Test_dfy= pd.read_csv("SyntheticData/Testdatay_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg),header=None,)
x_test =  Test_dfx.T.values.astype(np.float32)
y_test = Test_dfy.T.values.astype(np.float32)

train_df =  datawrapper( x_train,y_train)
valid_df =  datawrapper( x_valid,y_valid)
test_df =  datawrapper( x_test,y_test)

for seed in range(10):
    seed_all(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    train_dl = DataLoader(train_df, batch_size= batchsize,generator=g, num_workers=8)
    valid_dl = DataLoader(valid_df, batch_size= 125,num_workers=8)
    test_dl = DataLoader(test_df, batch_size= 2000, num_workers=8)



    shutil.rmtree(ckpt_dir,ignore_errors=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_regret",
        dirpath=ckpt_dir, 
        filename="model-{epoch:02d}-{val_loss:.2f}",
        mode="min",
    )
    
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir= log_dir, version=seed)
    trainer = pl.Trainer(max_epochs= max_epochs,callbacks=[checkpoint_callback],  min_epochs=5, logger=tb_logger)
    model = SPO(net=nn.Linear(5,40), lr= lr,l1_weight=l1_weight, seed=seed, max_epochs= max_epochs)
    trainer.fit(model, train_dl, valid_dl)
    best_model_path = checkpoint_callback.best_model_path
    model = SPO.load_from_checkpoint(best_model_path,
    net=nn.Linear(5,40), lr= lr, l1_weight=l1_weight, seed=seed)

    y_pred = model(torch.from_numpy(x_test).float()).squeeze()
    # pred_df = pd.DataFrame(y_pred.detach().numpy())
    regret_list = regret_aslist(spsolver, y_pred, torch.from_numpy(y_test).float())
    df = pd.DataFrame({"regret":regret_list})
    df.index.name='instance'
    df ['model'] = 'SPO'
    df['seed'] = seed
    df ['batchsize'] = batchsize
    df ['noise'] = noise
    df ['deg'] =  deg
    df['N'] = N

    df['l1_weight'] = l1_weight
    df['lr'] = lr
    with open(regretfile, 'a') as f:
        df.to_csv(f, header=f.tell()==0)


    ##### Summary
    result = trainer.test(model, dataloaders=test_dl)
    df = pd.DataFrame(result)
    df ['model'] = 'SPO'
    df['seed'] = seed
    df ['batchsize'] = batchsize
    df ['noise'] = noise
    df ['deg'] =  deg
    df['N'] = N
    df['l1_weight'] = l1_weight
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
df['model'] ='SPO'
df.to_csv("LearningCurve/SPO_data_N_{}_noise_{}_deg_{}.csv".format(N,noise,deg))