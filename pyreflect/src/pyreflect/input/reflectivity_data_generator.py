#This block is the class that is used to generate the SLD and NR curves
from typing import List, Tuple

import typer
from tqdm.auto import tqdm
from scipy.stats import qmc
# from scipy.interpolate import CubicSpline
from refl1d.names import *
import numpy as np
import os

class ReflectivityDataGenerator:
    """
        Generate pairs of NR, SLD profile curves using Reflectivity Model
    """
    def __init__(self,num_layers,layer_desc = None, layer_bound = None):
        self.num_layers = num_layers
        match self.num_layers:
            # Use physical model from original code
            case 5 if layer_desc is None and layer_bound is None:
                typer.echo(f"Using Legacy Reflectivity Model layer{num_layers}...\n")
                self.model = ReflectivityFiveLayerModel()
            case _:
                typer.echo(f"Using Model layer{num_layers}...\n")
                self.model = ReflectivityModel(num_layers,layer_desc=layer_desc,layer_bound=layer_bound)

    def generate(self,num_curves:int)-> Tuple[np.ndarray, np.ndarray]:
        """
        For each curve, sample a random set of parameters
            within the defined bounds for each layer,
            update the fixed model with these parameters, and
            pass the resulting model into calculate_reflectivity
            to compute the reflectivity curve.

        :param num_curves: int
        :return Tuple(np.ndarray,np.ndarray):
        """
        self.model.generate(num_curves)
        #add noise and log transform NR
        # pars, train_data = self.model.get_preprocessed_data()

        #standardize all sld profiles to start at same place
        for sld_arr in self.model._smooth_array:
            sld_arr[0] = np.array(sld_arr[0]) - np.min(sld_arr[0])

        nr_curves = []
        sld_curves = []
        for i in range(len(self.model._smooth_array)):
            nr_curves.append(np.array([self.model.q,self.model._refl_array[i]]))
            sld_curves.append(np.array(self.model._smooth_array[i]))

        return np.stack(nr_curves), np.stack(sld_curves)


def calculate_reflectivity(q, model_description, q_resolution=0.0294855):
    """
        Reflectivity calculation using refl1d
    """
    zeros = np.zeros(len(q))
    dq = q_resolution * q / 2.355

    # The QProbe object represents the beam
    probe = QProbe(q, dq, data=(zeros, zeros))

    layers = model_description['layers']
    sample = Slab(material=SLD(name=layers[0]['name'],
                               rho=layers[0]['sld']), interface=layers[0]['roughness'])
    # Add each layer
    for l in layers[1:]:
        sample = sample | Slab(material=SLD(name=l['name'],
                               rho=l['sld'], irho=l['isld']),
                               thickness=l['thickness'], interface=l['roughness'])

    probe.background = Parameter(value=model_description['background'], name='background')
    expt = Experiment(probe=probe, sample=sample)

    q, r = expt.reflectivity()
    z, sld, _ = expt.smooth_profile()
    #this makes all SLD curves have 900 datapoints per curve
    zNew = np.linspace(z[0], z[-1], num=900)
    sldNew = np.interp(zNew, z, sld)  # linear, no overshoot
    return model_description['scale'] * r, zNew, sldNew


class ReflectivityFiveLayerModel(object):
    """
    Reflectivity model with five layers, with substrate, and SiOx layer as base,
    polymer film layer in the middle and top air layer.
    """
        # Neutrons come in from the last item in the list
    def __init__(self,q=None, name='polymer'):

        self.parameters = [
                  dict(i=0, par='roughness', bounds=[1.177, 1.5215]),
                  # The following is the Si oxide layer
                  dict(i=1, par='sld', bounds=[3.47, 3.47]),
                  dict(i=1, par='thickness', bounds=[9.7216, 14.624]),
                  dict(i=1, par='roughness', bounds=[1.108, 2.998]),

                  # The next 5 layers are the polymer
                    dict(i=2, par='sld', bounds=[3.7235, 4.197]),
                  dict(i=2, par='thickness', bounds=[8.717,98.867]),
                  dict(i=2, par='roughness', bounds=[2.2571,38.969]),
                    dict(i=3, par='sld', bounds=[1.6417, 3.1033]),
                  dict(i=3, par='thickness', bounds=[117.4,239.91]),
                  dict(i=3, par='roughness', bounds=[19.32,95.202]),
                    dict(i=4, par='sld', bounds=[3.0246, 4.6755]),
                  dict(i=4, par='thickness', bounds=[64.482,94.768]),
                  dict(i=4, par='roughness', bounds=[15.713,28.007]),
                  dict(i=5, par='sld', bounds=[1.501, 4.9837]),
                  dict(i=5, par='thickness', bounds=[51.655,83.334]),
                  dict(i=5, par='roughness', bounds=[9.7741,25.373]),
                  dict(i=6, par='sld', bounds=[0.85516,4.4906]),
                  dict(i=6, par='thickness', bounds=[58.479, 86.738]),
                  dict(i=6, par='roughness', bounds=[43.155, 110.11]),
                 ]
        self.model_description = dict(layers=[
                                dict(sld=2.07, isld=0, thickness=0, roughness=1.8272, name='substrate'),
                                dict(sld=3.47, isld=0, thickness=10.085, roughness=1.108, name='siox'),
                                dict(sld=3.734, isld=0, thickness=8.717, roughness=2.2571, name='bottom'),
                                dict(sld=3.1033, isld=0, thickness=239.91, roughness=37.707, name='bottom'),
                                dict(sld=3.0246, isld=0, thickness=91.232, roughness=20.147, name='bulk'),
                                dict(sld=3.0246, isld=0, thickness=51.655, roughness=20.147, name='bulk'),
                                dict(sld=0.85516, isld=0, thickness=62.189, roughness=43.155, name='bottom'),
                                dict(sld=0, isld=0, thickness=0, roughness=0, name='air')
                         ],
                         scale=1,
                         background=0,
                        )
        # The following are unmodified physical parameters and corresponding reflectivity data
        self._pars_array = []
        self._refl_array = []
        # This creates the SLD curve list
        self._smooth_array = []
        # The following are the parameters, mapped between -1 and 1.
        self._train_pars = []
        self._train_data = None
        self._config_name = name

        if q is None:
            self.q = np.logspace(np.log10(0.008101436040354381), np.log10(0.1975709062238298), num=308)
        else:
            self.q = q

    @classmethod
    def from_dict(cls, pars, q_array=None):
        """
            Create ReflectivityModels object from a dict that
            defines the reflectivity model parameters and how
            the training set should be generated.
        """
        m = cls(q_array, name=pars['name'])
        m.model_description =  pars['model']
        m.parameters = pars['parameters']
        return m

    def generate(self, n=100):
        """
            For each curve, sample a random set of parameters
            within the defined bounds for each layer,
            update the fixed model with these parameters, and
            pass the resulting model into calculate_reflectivity
            to compute the reflectivity curve.
        """
        npars = len(self.parameters)
        self._train_pars = np.random.uniform(low=-1, high=1, size=[n, npars])
        # Compute model parameters and reflectivity using these values
        self.compute_reflectivity()

    def to_model_parameters(self, pars):
        """
            Transform an array of parameters to a list of calculable models
        """
        pars_array = np.zeros(pars.shape)

        for i, par in enumerate(self.parameters):
            a = (par['bounds'][1]-par['bounds'][0])/2.
            b = (par['bounds'][1]+par['bounds'][0])/2.
            pars_array.T[i] = pars.T[i] * a + b

        return pars_array

    def compute_reflectivity(self):
        """
            Transform an array of parameters to a list of calculable models
            and compute reflectivity
        """
        print("Computing reflectivity")
        self._pars_array = self.to_model_parameters(self._train_pars)
        # Compute reflectivity
        for p in tqdm(self._pars_array,desc="Generating reflectivity curves",colour="green"):
            _desc = self.get_model_description(p)
            r, z, sld = calculate_reflectivity(self.q, _desc)
            self._refl_array.append(r)
            self._smooth_array.append([z,sld])

    def get_model_description(self, pars):
        """
            Convert the parameter list to a model description that we can use
            to compute R(q).
        """
        for i, par in enumerate(self.parameters):
            self.model_description['layers'][par['i']][par['par']] = pars[i]
        return self.model_description

    def get_preprocessed_data(self, errors=None):
        """
            Pre-process data
            If errors is provided, a random error will be added, taking the errors array
            as a relative uncertainty.
        """
        if errors is None:

            self._train_data = np.log10(self._refl_array*self.q**2/self.q[0]**2)
        else:
            _data = self._refl_array * (1.0 + np.random.normal(size=len(errors)) * errors)
            # Catch the few cases where we generated a negative intensity and take
            # the absolute value
            _data[_data<0] = np.fabs(_data[_data<0])
            self._train_data = np.log10(_data*self.q**2/self.q[0]**2)

        return self._train_pars, self._train_data

    def save(self, output_dir=''):
        """
            Save all data relevant to a training set
            @param output_dir: directory used to store training sets
        """
        # Save q values
        np.save(os.path.join(output_dir, "%s_q_values" % self._config_name), self.q)

        # Save training set
        if self._train_data is not None:
            np.save(os.path.join(output_dir, "%s_data" % self._config_name), self._train_data)
            np.save(os.path.join(output_dir, "%s_pars" % self._config_name), self._train_pars)

    def load(self, data_dir=''):
        self.q = np.load(os.path.join(data_dir, "%s_q_values.npy" % self._config_name))
        self._train_data = np.load(os.path.join(data_dir, "%s_data.npy" % self._config_name))
        self._train_pars = np.load(os.path.join(data_dir, "%s_pars.npy" % self._config_name))
        return self.q, self._train_data, self._train_pars


class ReflectivityModel:
    """
        A revised version of the reflectivity model that has fixed 6 layers. This model support dynamic input layers.
    """
    def __init__(self,
                 num_layers: int,
                 layer_desc: List[dict] = None,
                 layer_bound: List[dict] = None,
                 q=None):

        self.q = np.asarray(q) if q is not None else np.logspace(np.log10(0.0081), np.log10(0.1975), num=308)
        self.num_layers = num_layers
        self.model_description = {
            "layers": layer_desc if layer_desc is not None else self._auto_generate_layer_description(),
            "scale": 1,
            "background": 0,
        }
        self.parameters = layer_bound if layer_bound is not None else self._auto_generate_layer_bounds()

        # The following are unmodified physical parameters and corresponding reflectivity data
        self._pars_array = []
        self._refl_array = []
        # This creates the SLD curve list
        self._smooth_array = []
        # The following are the parameters, mapped between -1 and 1.
        self._train_pars = []
        self._train_data = None

    def _auto_generate_layer_bounds(self):
        typer.echo("Generating random layer bounds...")
        bounds=[
            dict(i=0, par="roughness", bounds=[0, 5]),
            #siox layer
            dict(i=1,par="sld",bounds=[3,3.9]),
            dict(i=1,par="thickness",bounds=[10,25]),
            dict(i=1,par="roughness",bounds=[1,10])
        ]

        for i in range(self.num_layers):
            idx = i + 2  # layer indices after substrate and siox
            bounds.extend([
                dict(i=idx, par='sld', bounds=[0, 6.4]),
                dict(i=idx, par='thickness', bounds=[10, 300]),
                dict(i=idx, par='roughness', bounds=[0, 150])
            ])
        return bounds

    def _auto_generate_layer_description(self):
        typer.echo("Generating layer description based on layer bounds...")

        random_layers = [
                     dict(sld=2.07, isld=0, thickness=0, roughness=3, name='substrate'),
                     dict(sld=3.3, isld=0, thickness=16.2139, roughness=7, name='siox'),
                 ] + [
                     dict(
                         sld=np.random.uniform(0, 6.4),
                         isld=0,
                         thickness=np.random.randint(10, 300),
                         roughness=np.random.randint(0, 150),
                         name=f"layer{i}"
                     ) for i in range(self.num_layers)
                 ] + [
                     dict(sld=0.0, isld=0, thickness=0, roughness=0, name='air')
                 ]

        return random_layers

    def generate(self, n=100):
        npars = len(self.parameters)

        sampler = qmc.LatinHypercube(d=npars)
        sample = sampler.random(n=n) # shape: (n, npars)
        # self._train_pars = np.random.uniform(low=-1, high=1, size=[n, npars])
        self._train_pars = sample*2 -1 # map linearly to [-1, 1]
        self.compute_reflectivity()

    def to_model_parameters(self, pars):
        pars_array = np.zeros(pars.shape)
        for i, par in enumerate(self.parameters):
            a = (par['bounds'][1] - par['bounds'][0]) / 2.
            b = (par['bounds'][1] + par['bounds'][0]) / 2.
            pars_array.T[i] = pars.T[i] * a + b
        return pars_array

    def compute_reflectivity(self):
        self._pars_array = self.to_model_parameters(self._train_pars)
        for p in tqdm(self._pars_array,desc="Computing reflectivity for NR, SLD profile",colour='green'):
            _desc = self.get_model_description(p)
            r, z, sld = calculate_reflectivity(self.q, _desc)
            self._refl_array.append(r)
            self._smooth_array.append([z, sld])

    def get_model_description(self, pars):
        for i, par in enumerate(self.parameters):
            self.model_description['layers'][par['i']][par['par']] = pars[i]
        return self.model_description

