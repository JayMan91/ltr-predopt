from matching_models import *
import shutil
from pytorch_lightning.callbacks import ModelCheckpoint
exp = "Matching1"
params_dict = {"Matching1":{'p':0.25, 'q':0.25},"Matching2":{'p':0.5, 'q':0.5} }
params = params_dict[exp]
###################################### Hyperparams #########################################
lr = 1e-3
growth =1.
if __name__ == "__main__":
    x, y,m = get_cora()

    x_train, x_test = x[:22], x[22:]
    y_train, y_test = y[:22], y[22:]
    m_train, m_test = m[:22], m[22:]
    print("x_train",x_train.shape,"x test", x_test.shape,"m train",m_train.shape, "y train",y_train.shape)

    # print(solver(y_train[1], m_train[1], relaxation=False).shape)
    solpool_np = np.array([solver(y_train[i], m_train[i], relaxation=False) for i in range(len(y_train))])
    solpool_np = np.unique(solpool_np, axis=0)
    solpool =  torch.from_numpy(solpool_np).float()
    # print(solpool_np.shape)
    # solpool = torch.load('matching_initpool.pt')


    train_df = CoraDataset( x_train,y_train,m_train,params= params)
    valid_df = CoraDataset( x_test,y_test,m_test, params= params)

    train_dl = DataLoader(train_df, batch_size=4)
    valid_dl = DataLoader(valid_df, batch_size=5)
    for repeat in range(4):
        for alpha in [0]:

            shutil.rmtree("ckpt_dir/CombinedPO/matching2/",ignore_errors=True)
            checkpoint_callback = ModelCheckpoint(
                        monitor="val_regret",
                        dirpath="ckpt_dir/CombinedPO/matching2/",
                        filename="model-{epoch:02d}-{val_loss:.2f}",
                        save_top_k=1,save_last = True,
                        mode="min",
                    )
                    

            trainer = pl.Trainer(max_epochs= 40, callbacks=[checkpoint_callback], min_epochs=5)

            model = CombinedPO(alpha=alpha,loss_fn = Listnet_loss, solpool=solpool,growpool_fn =growpool_fn,lr= lr,growth=growth)
            trainer.validate(model, dataloaders=valid_dl)

            trainer.fit(model, train_dl, valid_dl)
            best_model_path = checkpoint_callback.best_model_path
            print("Model Path:",best_model_path)

            model = CombinedPO.load_from_checkpoint(best_model_path ,
            alpha=alpha,loss_fn = Listnet_loss, solpool=solpool,growpool_fn =growpool_fn,lr= lr,growth=growth)    
            result = trainer.test(dataloaders=valid_dl)
            df = pd.DataFrame(result)
            df ['model'] = 'Listwise'
            df['alpha'] = alpha
            df['growth'] = growth
            df['lr'] = lr
            with open("{}CombinedPORslt.csv".format(exp), 'a') as f:
                    df.to_csv(f, header=f.tell()==0)





