from ..input import DataProcessor

from ..config.runtime import DEVICE
import torch
from .cnn import CNN

from .model_trainer import ModelTrainer

# Training parameters
LEARNING_RATE = 2.15481e-05
WEIGHT_DECAY = 2.6324e-05
SPLIT_RATIO = 0.8

class NRSLDModelTrainer(ModelTrainer):
    def __init__(self,X,y,layers, batch_size, epochs,dropout=0.5):
        super().__init__(batch_size = batch_size, epochs = epochs)
        self.model = CNN(layers=layers,dropout_prob=dropout).to(DEVICE) #model
        self.X = X
        self.y = y
        self.layers = layers
        self.batch_size = batch_size
        self.epochs = epochs

    def train_pipeline(self):
        self.model.train()

        list_arrays = DataProcessor.split_arrays(self.X, self.y, size_split=SPLIT_RATIO)
        tensor_arrays = DataProcessor.convert_tensors(list_arrays)

        # train valid split
        train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = DataProcessor.get_dataloaders(
            *tensor_arrays, batch_size= self.batch_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_fn = torch.nn.MSELoss()

        # Training loop
        for epoch in range(self.epochs):
            train_loss = self.train_model(self.model, train_loader, optimizer, loss_fn)
            val_loss = self.validate_model(self.model, valid_loader, loss_fn)
            print(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

        return self.model


