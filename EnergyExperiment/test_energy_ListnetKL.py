import pytorch_lightning as pl
from PO_model import *
from run_energy import *
import sklearn
from sklearn.preprocessing import StandardScaler
from get_energy import get_energy
import shutil
###################################### Hyperparams #########################################
lr= 0.01


if __name__ == "__main__":
    ### training data prep
    x_train, y_train, x_test, y_test = get_energy(fname= 'prices2013.dat')
    x_train = x_train[:,1:]
    x_test = x_test[:,1:]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_train = x_train.reshape(-1,48,x_train.shape[1])
    y_train = y_train.reshape(-1,48)
    x_test = x_test.reshape(-1,48,x_test.shape[1])
    y_test = y_test.reshape(-1,48)
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train,y_test), axis=0)
    x,y = sklearn.utils.shuffle(x,y,random_state=0)
    x_train, y_train = x[:552], y[:552]
    x_valid, y_valid = x[552:652], y[552:652]
    x_test, y_test = x[652:], y[652:]
    print(x_train.shape, x_valid.shape, x_test.shape)
    test_df = EnergyDatasetWrapper( x_test,y_test,param, relax=False)
    test_dl = data_utils.DataLoader(test_df, batch_size= 24)

    train_df = EnergyDatasetWrapper( x_train,y_train,param, relax=False)

    train_dl = data_utils.DataLoader(train_df, batch_size=len(y))
    for x_train,y_train,sol_train in train_dl:
        solpool_np = np.unique(sol_train.detach().numpy(), axis=0)
        solpool =  torch.from_numpy(solpool_np).float()

    train_dl = DataLoader(train_df, batch_size= 24)

    valid_df = EnergyDatasetWrapper( x_valid,y_valid,param, relax=False)
    valid_dl = data_utils.DataLoader(valid_df, batch_size= 24)

    test_df = EnergyDatasetWrapper( x_test,y_test,param, relax=False)
    test_dl = data_utils.DataLoader(test_df, batch_size= 24)



    
    for repeat in range(6):
        for growth in [1.]:
            shutil.rmtree('ckpt_dir/Listnet/Energy1/',ignore_errors=True)
            checkpoint_callback = ModelCheckpoint(
                monitor="val_regret",
                dirpath="ckpt_dir/Listnet/Energy1/",
                filename="model-{epoch:02d}-{val_loss:.2f}",
                save_top_k=2,save_last = True,
                mode="min",
            )
            trainer = pl.Trainer(max_epochs= 20, callbacks=[checkpoint_callback], min_epochs=5)
            model = SemanticPO(loss_fn = Listnet_KLloss,regret_fn= regret_fn,solpool=solpool,growpool_fn =growpool_fn,growth=growth,lr= lr)


            trainer.fit(model, train_dl, valid_dl)
            best_model_path = checkpoint_callback.best_model_path
            model = SemanticPO.load_from_checkpoint(best_model_path ,
                loss_fn = Listnet_KLloss,regret_fn= regret_fn,solpool=solpool,growpool_fn =growpool_fn,growth=growth,lr= lr)


            result = trainer.test(dataloaders=test_dl)
            df = pd.DataFrame(result)
            df ['model'] = 'Listnet(KL)'
            df['growth'] = growth
            df['lr'] = lr
            
            with open("Energy1ListwiseKLRslt.csv", 'a') as f:
                df.to_csv(f, header=f.tell()==0)