import numpy as np
from sklearn.model_selection import train_test_split
import torch

class DataProcessor:
    def __init__(self, seed=42):
        """
        Base class for processing data.
        """
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    @staticmethod
    def split_arrays(X, y, size_split=0.8):
        """
        Splits the dataset into training, validation, and test sets.
        """
        crv_tr, crv_hld, chi_tr, chi_hld = train_test_split(X, y, train_size=size_split)
        crv_val, crv_tst, chi_val, chi_tst = train_test_split(crv_hld, chi_hld, test_size=0.5)

        return [crv_tr, chi_tr, crv_val, chi_val, crv_tst, chi_tst]

    @staticmethod
    def convert_tensors(list_arrays):
        # Convert single Numpy array to tensors dtype float32
        if isinstance(list_arrays, np.ndarray):
            return torch.from_numpy(list_arrays).float()

        # Convert List of Numpy array to tensors
        tensor_arrays = [torch.from_numpy(array).float() for array in list_arrays]
        return tensor_arrays

    @staticmethod
    def get_dataloaders(xtrain, ytrain, xval, yval, xtest, ytest, batch_size=32):
        """
        Converts split datasets into PyTorch dataloaders.
        """
        train_dataset = torch.utils.data.TensorDataset(xtrain,ytrain)
        valid_dataset = torch.utils.data.TensorDataset(xval,yval)
        test_dataset = torch.utils.data.TensorDataset(xtest,ytest)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return [train_dataset,valid_dataset,test_dataset,train_loader, valid_loader, test_loader]

    @staticmethod
    def normalize_xy_curves(curves, apply_log=False, min_max_stats=None) -> np.ndarray:
        """
        Normalize NR or SLD curves using global min/max or provided stats.

        Args:
            curves (np.ndarray): Input array of shape (N, 2, L).
            apply_log (bool): Whether to apply log10 transformation on y.
            min_max_stats (dict or None): Optional stats dict {'x': {'min', 'max'}, 'y': {'min', 'max'}}.

        Returns:
            Tuple of normalized curves and used min_max_stats.
        """
        curves = np.array(curves)
        assert curves.ndim == 3 and curves.shape[1] == 2, \
            f"Expected shape (N, 2, L), got {curves.shape}"

        if apply_log:
            curves[:, 1, :] = np.log10(np.clip(curves[:, 1, :], 1e-8, None))

        x_points = curves[:, 0, :]
        y_points = curves[:, 1, :]

        # Use existing stats if provided
        if min_max_stats:
            min_valXNR = min_max_stats['x']['min']
            max_valXNR = min_max_stats['x']['max']
            min_valYNR = min_max_stats['y']['min']
            max_valYNR = min_max_stats['y']['max']
        else:
            min_valXNR = np.min(x_points)
            max_valXNR = np.max(x_points)
            min_valYNR = np.min(y_points)
            max_valYNR = np.max(y_points)

        # Normalize
        x_points = (x_points - min_valXNR) / (max_valXNR - min_valXNR)
        y_points = (y_points - min_valYNR) / (max_valYNR - min_valYNR)

        normalized_curves = np.stack([x_points, y_points], axis=1)

        return normalized_curves

    @staticmethod
    def get_xy_norm_stats(self,data_curves):
        curves = np.array(data_curves)

        assert curves.ndim == 3 and curves.shape[1] == 2, \
            f"Expected shape (N, 2, L), got {curves.shape}"

        x_points = curves[:, 0, :]
        y_points = curves[:, 1, :]

        min_valXNR = np.min(x_points)
        max_valXNR = np.max(x_points)
        min_valYNR = np.min(y_points)
        max_valYNR = np.max(y_points)

        # norm stats
        return {
            'x':{
                'min': min_valXNR,
                'max': max_valXNR
            },
            'y':{
                'min': min_valYNR,
                'max': max_valYNR
            }
        }


    @staticmethod
    def denormalize_xy_curves(norm_curves, stats, apply_exp=False):
        """
        Denormalize curves from normalized form back to original scale.

        Args:
            norm_curves (np.ndarray): shape [N, 2, L]
            stats (dict): min/max values from normalization
            apply_exp (bool): If data was log-scaled, apply 10** to get back

        Returns:
            np.ndarray: denormalized curves
        """
        x_norm = norm_curves[:, 0, :]
        y_norm = norm_curves[:, 1, :]

        # Denormalize
        x_orig = x_norm * (stats['x']['max'] - stats['x']['min']) + stats['x']['min']
        y_orig = y_norm * (stats['y']['max'] - stats['y']['min']) + stats['y']['min']

        if apply_exp:
            y_orig = np.power(10, y_orig)

        return np.stack([x_orig, y_orig], axis=1)

class SLDChiDataProcessor(DataProcessor):
    def __init__(self, expt_sld_file_path, sld_file_path, chi_params_file_path, seed=42):
        """
        Processes SLD and Chi parameter data.
        """
        super().__init__(seed)
        self.sld_file_path = sld_file_path
        self.chi_params_file_path = chi_params_file_path
        self.expt_sld_file_path = expt_sld_file_path
        self.sld_arr = None
        self.expt_arr = None
        self.params_arr = None
        self.num_params = None

    def load_data(self):
        """Loads SLD and Chi parameter data."""
        self.sld_arr = np.load(self.sld_file_path,allow_pickle=True)
        self.params_arr = np.load(self.chi_params_file_path,allow_pickle=True)
        self.expt_arr = np.load(self.expt_sld_file_path,allow_pickle=True)
        self.num_params = self.params_arr.shape[1]

    @classmethod
    def remove_flat_samples(cls, data_arr):
        # Remove flat data
        flat_data = []
        for i in range(data_arr.shape[0]):
            y_start = data_arr[i, 1, 0]
            if data_arr[i, 1, 1] == y_start and data_arr[i, 1, 2] == y_start:
                flat_data.append(i)

        return np.delete(data_arr,flat_data,axis=0), flat_data

    def preprocess_data(self):
        """Cleans and normalizes data by removing flat and non-impact data."""

        assert self.sld_arr is not None or self.params_arr is not None and self.expt_arr is not None, "data not loaded."

        # Remove flat data
        self.sld_arr,flat_data = self.remove_flat_samples(self.sld_arr)
        self.params_arr = np.delete(self.params_arr, flat_data, 0)

        # Remove non-impact data (chi1 range filter)
        bad_chi1 = [i for i in range(self.sld_arr.shape[0]) if not (0.07 <= self.params_arr[i, 0] <= 0.12)]
        self.sld_arr = np.delete(self.sld_arr, bad_chi1, 0)
        self.params_arr = np.delete(self.params_arr, bad_chi1, 0)

        # Batch normalize chi parameters
        for i in range(self.params_arr.shape[1]):
            min_val, max_val = self.params_arr[:, i].min(), self.params_arr[:, i].max()
            if max_val > min_val:
                self.params_arr[:, i] = ((self.params_arr[:, i] - min_val) * 2 / (max_val - min_val)) - 1

        return self.sld_arr, self.params_arr

class NRSLDDataProcessor(DataProcessor):
    def __init__(self, nr_file_path=None, sld_file_path=None, seed=42):
        """
        Processes NR and SLD curve data.
        """
        super().__init__(seed)
        self.nr_file_path = nr_file_path
        self.sld_file_path = sld_file_path
        self._nr_arr = None
        self._sld_arr = None

    def load_data(self,new_nr_file=None, new_sld_file=None):
        """

        :param new_nr_file:
        :param new_sld_file:
        :return:

        :raise
        - OSError
        """
        # Decide file paths: use new ones if provided, else use those from init
        nr_path = new_nr_file if new_nr_file is not None else self.nr_file_path
        sld_path = new_sld_file if new_sld_file is not None else self.sld_file_path

        if nr_path:
            self._nr_arr = np.load(nr_path,allow_pickle=True)

        if sld_path:
            self._sld_arr = np.load(sld_path,allow_pickle=True)

        return self

    def normalize_nr(self, norm_stats):
        """Normalizes NR curves."""
        if self._nr_arr is None:
            raise FileNotFoundError(f"NR file not loaded from path:{self.nr_file_path}")

        if norm_stats is None:
            raise Exception("norm_stats is none.")

        self._nr_arr,_ = DataProcessor.normalize_xy_curves(self._nr_arr, apply_log=True, min_max_stats= norm_stats)
        self.norm_stats = norm_stats

        return self._nr_arr

    def normalize_sld(self, norm_stats =None):
        """Normalizes SLD curves."""
        if self._sld_arr is None:
            raise FileNotFoundError(f"SLD File not loaded from path: {self.sld_file_path}")

        if norm_stats is not None:
            self._sld_arr,_ = DataProcessor.normalize_xy_curves(self._sld_arr, apply_log=False, min_max_stats=norm_stats)
        else:
            self._sld_arr,self.norm_stats['sld'] = DataProcessor.normalize_xy_curves(self._sld_arr,apply_log=False)

        return self._sld_arr

    def get_normalization_stats(self):
        """
        Retrieve the min and max values used during normalization for NR and SLD data.

        :return: Normalization statistics.
        :rtype: dict
        :example:
            {
                "nr": {
                    "x": {"min": float, "max": float},
                    "y": {"min": float, "max": float}
                },
                "sld": {
                    "x": {"min": float, "max": float},
                    "y": {"min": float, "max": float}
                }
            }
        """
        return self.norm_stats

    def reshape_nr_to_single_channel(self,nr_data:np.ndarray)->np.ndarray:
        """
        Reshapes NR data to (batch_size, 1, sequence_length) for CNN input.
        Remove the momentum q range(y-axis).

        Returns:
        - reshaped_nr (numpy.ndarray): Reshaped NR data.
        """
        if not isinstance(nr_data, np.ndarray):
            raise TypeError("nr_data must be a numpy array.")

        if nr_data.ndim != 3 or nr_data.shape[1] != 2:
            raise ValueError(f"Expected input shape (batch_size, 2, sequence_length), got {nr_data.shape}")

        # Selecting the second channel (index 1) and reshaping to (batch_size, 1, sequence_length)
        reshaped_nr = nr_data[:, 1][:, np.newaxis, :]
        return reshaped_nr

    def denormalize(self,normalized_curves: np.ndarray, curve_type:str,min_max_stats):
        """
        Denormalize NR or SLD curves using saved normalization metadata.

        :param normalized_curves: np.ndarray:shape [N, 2, L]
        :param curve_type: 'nr' or 'sld'
        :param min_max_stats: min max for de-normalization
        :return: np.ndarray:shape [N, 2, L]
        """
        apply_exp = False

        match curve_type:
            case 'nr':
                apply_exp = True
            case 'sld':
                apply_exp = False
            case _:
                raise ValueError(f"Invalid curve type {curve_type}")

        if min_max_stats is None:
            raise ValueError("Not found normalization statistics.")

        return self.denormalize_xy_curves(normalized_curves, stats=min_max_stats, apply_exp= apply_exp)

