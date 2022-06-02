from matching_models import *
def model_tune(config,train_dl, valid_dl, solpool=None,num_epochs=30, num_gpus=0):
    ##### ***** Model Specific name and parameter *****
    model = SemanticPO(loss_fn = pairwise_loss, solpool=solpool,growpool_fn =growpool_fn,lr= config['lr'],margin= config['margin'])
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus= num_gpus,
        callbacks=[
            TuneReportCallback(
                {
                    "regret": "ptl/val_regret",
                    "mse": "ptl/val_mse"
                },
                on="validation_end")
        ])
    trainer.fit(model,train_dl, valid_dl)
    

def tune_model_asha(train_dl, valid_dl,solpool=None,num_samples=1, num_epochs=20, gpus_per_trial=0):
    ### ***** Model Specific config *****
    config = {
            "lr": tune.grid_search([0.005]),"margin":tune.grid_search([50])
        }
    scheduler = ASHAScheduler(
         time_attr='training_iteration',
            max_t=num_epochs,
            grace_period=8,
            reduction_factor=2)

    reporter = CLIReporter(
        ### ***** Model Specific parameter *****
            parameter_columns=["lr",'margin' ],
            metric_columns=[ "training_iteration","mse", "regret"])
    analysis = tune.run(
            tune.with_parameters(
                model_tune,train_dl = train_dl, valid_dl = valid_dl,solpool=solpool,
                num_epochs=num_epochs,
                num_gpus=gpus_per_trial),
            resources_per_trial={
                "cpu": 4,
                "gpu": gpus_per_trial
            },
            metric="regret",
            mode="min",
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            ### ***** Model Specific name *****
            name="tune_pairwiseMatching_asha")
    best_trial = analysis.get_best_trial("regret", "min", "last")
    print("Best trial final validation regret: {} mse: {}".format(
        best_trial.last_result["regret"], best_trial.last_result["mse"]))
    print("Best trial: lr- {}  margin {} final epoch- {}".format( best_trial.config["lr"], best_trial.config["margin"],
        best_trial.last_result["training_iteration"]))



if __name__ == "__main__":
    x, y,m = get_cora()

    x_train, x_test = x[:22], x[22:]
    y_train, y_test = y[:22], y[22:]
    m_train, m_test = m[:22], m[22:]
    print("x_train",x_train.shape,"x test", x_test.shape,"m train",m_train.shape, "y train",y_train.shape)

    print(solver(y_train[1], m_train[1], relaxation=False).shape)
    solpool_np = np.array([solver(y_train[i], m_train[i], relaxation=False) for i in range(len(y_train))])
    solpool_np = np.unique(solpool_np, axis=0)
    solpool =  torch.from_numpy(solpool_np).float()
    print(solpool_np.shape)


    train_df = CoraDataset( x_train,y_train,m_train,params={'p':0.25, 'q':0.25})
    valid_df = CoraDataset( x_test,y_test,m_test, params={'p':0.25, 'q':0.25})

    train_dl = DataLoader(train_df, batch_size=4)
    valid_dl = DataLoader(valid_df, batch_size=5)
    tune_model_asha(train_dl,valid_dl,solpool)

    # trainer = pl.Trainer(max_epochs= 25,log_every_n_steps=1)

    # model = SPO(lr=1e-2)

    # trainer.fit(model, train_dl, valid_dl)
    # result = trainer.test(dataloaders=valid_dl)
    # print(result)

