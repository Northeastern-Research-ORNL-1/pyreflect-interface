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

export interface ExportPngs {
  encoding: 'base64';
  normal: Record<string, string>;
  expanded: Record<string, string>;
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
  timing?: {
    generation?: number;
    training?: number;
    inference?: number;
    total?: number;
  };
  export_pngs?: ExportPngs;
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
  access_granted?: boolean;
  limit_source?: 'local_dev' | 'whitelist' | 'production';
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
export type DataSource = 'synthetic' | 'real';
export type Workflow = 'nr_sld' | 'sld_chi' | 'nr_sld_chi';
export type NrSldMode = 'train' | 'infer';
export type GpuTier = 'T4' | 'L4' | 'A10G' | 'L40S' | 'A100' | 'A100-80GB' | 'H100' | 'H200' | 'B200';

export const GPU_TIERS: { value: GpuTier; label: string; description: string; speed: string }[] = [
  { value: 'T4', label: 'NVIDIA T4', description: '$0.59/hr, 16GB, 65 TF', speed: '1×' },
  { value: 'L4', label: 'NVIDIA L4', description: '$0.80/hr, 24GB, 121 TF', speed: '2×' },
  { value: 'A10G', label: 'NVIDIA A10G', description: '$1.10/hr, 24GB, 125 TF', speed: '2×' },
  { value: 'L40S', label: 'NVIDIA L40S', description: '$1.95/hr, 48GB, 362 TF', speed: '5.5×' },
  { value: 'A100', label: 'NVIDIA A100', description: '$2.10/hr, 40GB, 312 TF', speed: '5×' },
  { value: 'A100-80GB', label: 'NVIDIA A100-80', description: '$2.50/hr, 80GB, 312 TF', speed: '5×' },
  { value: 'H100', label: 'NVIDIA H100', description: '$3.95/hr, 80GB, 989 TF', speed: '15×' },
  { value: 'H200', label: 'NVIDIA H200', description: '$4.54/hr, 141GB, 989 TF', speed: '15×' },
  { value: 'B200', label: 'NVIDIA B200', description: '$6.25/hr, 192GB, 2250 TF', speed: '35×' },
];
export type UploadRole =
  | 'auto'
  | 'nr_train'
  | 'sld_train'
  | 'experimental_nr'
  | 'normalization_stats'
  | 'nr_sld_model'
  | 'sld_chi_experimental_profile'
  | 'sld_chi_model_sld_file'
  | 'sld_chi_model_chi_params_file';
