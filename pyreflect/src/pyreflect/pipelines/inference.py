import torch
from ..models.cnn import CNN
from ..config.runtime import DEVICE

class Predictor:
    def __init__(self, spec):
        self.spec = spec

    def load_model(self, model_path):
        model = CNN(layers=self.spec.layers,dropout_prob=self.spec.dropout_prob)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        return model

    def predict(self, X):
        # TODO:
        pass