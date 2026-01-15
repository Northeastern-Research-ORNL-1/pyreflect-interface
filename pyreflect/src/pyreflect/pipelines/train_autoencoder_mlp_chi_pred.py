from ..input import DataProcessor
from ..models.chi_pred_model_trainer import ChiPredModelTrainer as ModelTrainer
from ..models import mlp, autoencoder as ae
from ..config.runtime import DEVICE
import torch

def run_model_training(
        X, y,
        latent_dim, batch_size,
        ae_epochs, mlp_epochs):

    sld_arr = X
    params_arr = y

    #Flatten for VAE takes shape [batch_size, 2 * features]
    sld_arr_cut_flat = sld_arr.reshape(sld_arr.shape[0], -1)

    # training, validation, testing arrays
    list_arrays = DataProcessor.split_arrays(sld_arr_cut_flat, params_arr, size_split=0.7)
    tensor_arrays = DataProcessor.convert_tensors(list_arrays)
    #train val data tensors from data processing
    tr_data, val_data,tst_data, tr_load, val_load,tst_load = DataProcessor.get_dataloaders(*tensor_arrays,batch_size)

    #The first sample input
    x,_ = tr_data[0]

    #The linear dimension of first input
    input_dim = x.numel()

    #number of Chi parameters
    num_params = params_arr.shape[1]

    # Initialize Model Trainer
    trainer = ModelTrainer(
        autoencoder=ae.VariationalAutoencoder(input_dim, latent_dim).to(DEVICE),
        mlp=mlp.deep_MLP(latent_dim, num_params).to(DEVICE),
        batch_size=batch_size,
        ae_epochs=ae_epochs,
        mlp_epochs= mlp_epochs,
        loss_fn=torch.nn.MSELoss(),
        latent_dim=latent_dim,
        num_params=num_params,
    )

    # Train Autoencoder
    trainer.train_autoencoder(tr_load, val_load)

    # Extract Latent Vectors & Store in DataFrame
    df_encoded_samples = trainer.extract_latent_vectors(tr_data)

    # Prepare MLP Datasets
    mlp_tr_data, mlp_val_data, mlp_tst_data, mlp_tr_load, mlp_val_load, mlp_tst_load = trainer.prepare_mlp_data(tr_load,
                                                                                                                val_load,                                                                                                   tst_load)
    # Train MLP
    trainer.train_mlp(mlp_tr_load, mlp_val_load)

    # Evaluate MLP on Training and Testing Data
    # df_train_samples = trainer.evaluate_model(mlp_tr_data, trainer.mlp)
    df_test_samples = trainer.evaluate_model(mlp_tst_data, trainer.mlp)

    # Calculate Prediction Errors
    df_l2_err = trainer.calculate_error(df_test_samples)

    # Print Final Results
    print("\nFinal Mean Prediction Errors:")
    print(df_l2_err)

    return trainer.mlp,trainer.autoencoder

