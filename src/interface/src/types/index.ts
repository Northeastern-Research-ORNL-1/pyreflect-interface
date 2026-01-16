export interface FilmLayer {
  name: string;
  sld: number;
  isld: number;
  thickness: number;
  roughness: number;
}

export interface GeneratorParams {
  numCurves: number;
  numFilmLayers: number;
}

export interface TrainingParams {
  batchSize: number;
  epochs: number;
  layers: number;
  dropout: number;
  latentDim: number;
  aeEpochs: number;
  mlpEpochs: number;
}

export interface NRData {
  q: number[];
  groundTruth: number[];
  computed: number[];  // NR computed from predicted SLD
}

export interface SLDData {
  z: number[];
  groundTruth: number[];
  predicted: number[];
}

export interface TrainingData {
  epochs: number[];
  trainingLoss: number[];
  validationLoss: number[];
}

export interface ChiData {
  x: number;
  predicted: number;
  actual: number;
}

export interface Metrics {
  mse: number;
  r2: number;
  mae: number;
}

export interface GenerateResponse {
  nr: NRData;
  sld: SLDData;
  training: TrainingData;
  chi: ChiData[];
  metrics: Metrics;
  name?: string;
  model_id?: string;
  model_size_mb?: number;
}

export interface Limits {
  max_curves: number;
  max_film_layers: number;
  max_batch_size: number;
  max_epochs: number;
  max_cnn_layers: number;
  max_dropout: number;
  max_latent_dim: number;
  max_ae_epochs: number;
  max_mlp_epochs: number;
}

export interface LimitsResponse {
  production: boolean;
  limits: Limits;
}

// Default limits (local/unlimited)
export const DEFAULT_LIMITS: Limits = {
  max_curves: 100000,
  max_film_layers: 20,
  max_batch_size: 512,
  max_epochs: 1000,
  max_cnn_layers: 20,
  max_dropout: 0.9,
  max_latent_dim: 128,
  max_ae_epochs: 500,
  max_mlp_epochs: 500,
};
