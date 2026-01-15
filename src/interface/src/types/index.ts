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
}
