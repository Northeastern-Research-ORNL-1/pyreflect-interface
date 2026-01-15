import torch
import pandas as pd
from . import autoencoder as ae
from ..input.data_processor import DataProcessor
from . import mlp as mlp_module
from ..config.runtime import DEVICE as device

class ChiPredModelTrainer:
    def __init__(self, autoencoder, mlp, batch_size, ae_epochs, mlp_epochs, loss_fn, latent_dim,
                 num_params):
        self.autoencoder = autoencoder
        self.mlp = mlp
        self.batch_size = batch_size
        self.ae_epochs = ae_epochs
        self.mlp_epochs = mlp_epochs
        self.loss_fn = loss_fn
        self.latent_dim = latent_dim
        self.num_params = num_params
        self.trained = False  # Track if models are trained

    def train_autoencoder(self, tr_load, val_load):
        """Train the autoencoder model and store losses."""
        print("\nTraining Autoencoder...")
        self.ae_train_loss, self.ae_val_loss = ae.train(self.autoencoder, tr_load, val_load,
                                                                      self.ae_epochs, self.loss_fn)
        self.trained = True

    def extract_latent_vectors(self, tr_data):
        """Extract latent vectors from the trained Autoencoder and store them in a Pandas DataFrame."""
        assert self.trained, "Autoencoder must be trained before extracting latent vectors."

        print("\nExtracting Latent Vectors...")
        encoded_samples = []
        self.autoencoder.eval()

        with torch.no_grad():
            for sample in tr_data:
                curve = sample[0].view(sample[0].size(0), -1).flatten().to(device)
                labels = sample[1]
                latent_vars = self.autoencoder.encoder(curve).cpu().numpy()

                latent_sample = {f"l{i + 1}": var for i, var in enumerate(latent_vars)}
                for i in range(self.num_params):
                    label_index = f'chi{i + 1}'
                    latent_sample[label_index] = labels[i].item()

                encoded_samples.append(latent_sample)

        return pd.DataFrame(encoded_samples)

    def get_full_output(self, ae_model, dataloader):
        """Run the Autoencoder on a dataset and return tensors for curves, reconstructions, latent space, and parameters."""
        list_curves, list_recon, list_latent, list_labels = [], [], [], []

        ae_model.to(device)
        ae_model.eval()
        with torch.no_grad():
            for curve, labels in dataloader:
                curve, labels = curve.to(device), labels.to(device)  # Move data to the correct device

                latent_vars = ae_model.encoder(curve.view(curve.size(0), -1))
                recon_curve = ae_model.decoder(latent_vars)

                list_curves.append(curve)
                list_labels.append(labels)
                list_latent.append(latent_vars)
                list_recon.append(recon_curve)

        return [
            torch.cat(list_curves).to(device),
            torch.cat(list_recon).to(device),
            torch.cat(list_latent).to(device),
            torch.cat(list_labels).to(device),
        ]

    def prepare_mlp_data(self, tr_load, val_load, tst_load):
        """Prepare MLP input datasets using latent space vectors and chi parameters."""
        print("\nPreparing MLP Training Data...")
        full_data_train = self.get_full_output(self.autoencoder, tr_load)
        full_data_valid = self.get_full_output(self.autoencoder, val_load)
        full_data_test = self.get_full_output(self.autoencoder, tst_load)

        mlp_input_data = [
            full_data_train[2], full_data_train[3],
            full_data_valid[2], full_data_valid[3],
            full_data_test[2], full_data_test[3]
        ]

        return DataProcessor.get_dataloaders(*mlp_input_data, self.batch_size)

    def train_mlp(self, mlp_tr_load, mlp_val_load):
        """Train the MLP model and store losses."""
        print("\nTraining MLP...")
        self.train_loss, self.val_loss = mlp_module.train(self.mlp, mlp_tr_load, mlp_val_load, self.mlp_epochs,
                                                        self.loss_fn)

    def evaluate_model(self, mlp_data, model):
        """Run the trained MLP on a dataset and store predictions in a Pandas DataFrame."""
        print("\nEvaluating MLP Model...")
        samples = []
        model.eval()

        with torch.no_grad():
            for sample in mlp_data:
                img = sample[0].view(sample[0].size(0), -1).flatten().to(device)
                label_val = sample[1]

                out_label = model(img).cpu().numpy()
                sample_dict = {f"pred_chi{i + 1}": enc for i, enc in enumerate(out_label)}

                for i in range(self.num_params):
                    label_index = f'chi{i + 1}'
                    sample_dict[label_index] = float(label_val[i])

                samples.append(sample_dict)

        return pd.DataFrame(samples)

    def calculate_error(self, df_samples):
        """Calculate percentage error in chi predictions."""
        print("\nCalculating Prediction Errors...")
        for i in range(self.num_params):
            df_samples[f'chi{i + 1}_err'] = (abs(df_samples[f'pred_chi{i + 1}'] - df_samples[f'chi{i + 1}']) /
                                             df_samples[f'chi{i + 1}']) * 100

        df_errors = pd.DataFrame(
            {f'chi{i + 1}_err': [df_samples[f'chi{i + 1}_err'].mean()] for i in range(self.num_params)})
        return df_errors
