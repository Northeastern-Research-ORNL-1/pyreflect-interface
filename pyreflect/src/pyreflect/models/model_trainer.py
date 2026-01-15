from abc import abstractmethod
import torch
from ..config.runtime import DEVICE

class ModelTrainer:
    def __init__(self,batch_size, epochs):
        self.batch_size = batch_size
        self.epochs = epochs

    @abstractmethod
    def train_pipeline(self):
        pass

    @staticmethod
    def train_model(model, train_loader, optimizer, loss_fn):
        model.train().to(DEVICE)
        total_loss = 0

        for data, label in train_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    @staticmethod
    def validate_model(model:torch.nn.Module, valid_loader, loss_fn):
        model.eval().to(DEVICE)
        total_loss = 0

        with torch.no_grad():
            for data, label in valid_loader:
                data, label = data.to(DEVICE), label.to(DEVICE)
                output = model(data)
                loss = loss_fn(output, label)
                total_loss += loss.item()

        return total_loss / len(valid_loader)
