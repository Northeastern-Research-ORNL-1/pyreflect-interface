'use client';

import { useState, useRef, type ChangeEvent, type DragEvent } from 'react';
import { FilmLayer, GeneratorParams, TrainingParams, Limits, DEFAULT_LIMITS } from '@/types';
import EditableValue from './EditableValue';
import styles from './ParameterPanel.module.css';

interface BackendStatus {
  pyreflect_available: boolean;
  has_settings: boolean;
  data_files: string[];
  curve_files: string[];
  expt_files: string[];
}

export interface ParameterPanelProps {
  filmLayers: FilmLayer[];
  generatorParams: GeneratorParams;
  trainingParams: TrainingParams;
  onFilmLayersChange: (layers: FilmLayer[]) => void;
  onGeneratorParamsChange: (params: GeneratorParams) => void;
  onTrainingParamsChange: (params: TrainingParams) => void;
  onGenerate: () => void;
  onReset: () => void;
  onUploadFiles: (files: File[]) => void;
  isGenerating: boolean;
  isUploading: boolean;
  backendStatus: BackendStatus | null;
  limits?: Limits;
  isProduction?: boolean;
}

export default function ParameterPanel({
  filmLayers,
  generatorParams,
  trainingParams,
  onFilmLayersChange,
  onGeneratorParamsChange,
  onTrainingParamsChange,
  onGenerate,
  onReset,
  onUploadFiles,
  isGenerating,
  isUploading,
  backendStatus,
  limits = DEFAULT_LIMITS,
  isProduction = false,
}: ParameterPanelProps) {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [expandedLayers, setExpandedLayers] = useState(() =>
    new Set<number>(filmLayers.map((_, index) => index))
  );

  const toggleLayer = (index: number) => {
    setExpandedLayers((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  const collapseAllLayers = () => {
    setExpandedLayers(new Set());
  };

  const expandAllLayers = () => {
    setExpandedLayers(new Set(filmLayers.map((_, index) => index)));
  };
  
  const updateLayer = (index: number, field: keyof FilmLayer, value: number | string) => {
    const newLayers = [...filmLayers];
    newLayers[index] = { ...newLayers[index], [field]: value };
    onFilmLayersChange(newLayers);
  };

  const addLayer = () => {
    const newLayer: FilmLayer = {
      name: `layer_${filmLayers.length - 1}`,
      sld: 3.0,
      isld: 0,
      thickness: 100,
      roughness: 20,
    };
    // Insert before 'air' layer (last layer)
    const newLayers = [...filmLayers];
    newLayers.splice(filmLayers.length - 1, 0, newLayer);
    onFilmLayersChange(newLayers);
    setExpandedLayers((prev) => {
      const next = new Set(prev);
      next.add(filmLayers.length - 1);
      return next;
    });
  };

  const removeLayer = (index: number) => {
    if (filmLayers.length <= 3) return; // Keep at least substrate, one layer, air
    const newLayers = filmLayers.filter((_, i) => i !== index);
    onFilmLayersChange(newLayers);
    setExpandedLayers((prev) => {
      const next = new Set<number>();
      newLayers.forEach((_, i) => {
        if (prev.has(i >= index ? i + 1 : i)) {
          next.add(i);
        }
      });
      return next;
    });
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    setSelectedFiles((prev) => [...prev, ...files]);
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragActive(true);
  };

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragActive(false);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragActive(false);
    const files = Array.from(event.dataTransfer.files);
    setSelectedFiles((prev) => [...prev, ...files]);
  };

  const handleDropzoneClick = () => {
    fileInputRef.current?.click();
  };

  const removeSelectedFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUpload = () => {
    if (selectedFiles.length === 0) return;
    onUploadFiles(selectedFiles);
    setSelectedFiles([]);
  };

  return (
    <div className={styles.panel}>
      {/* Film Layers Section */}
      <div className="section">
        <div className="section__header">
          <h3 className="section__title">Film Layers</h3>
          <div className={styles.layerActions}>
            <button
              className="btn btn--outline"
              onClick={expandedLayers.size ? collapseAllLayers : expandAllLayers}
              type="button"
              style={{ height: '28px', padding: '0 12px', fontSize: '11px' }}
            >
              {expandedLayers.size ? 'COLLAPSE' : 'EXPAND'}
            </button>
            <button
              className="btn btn--outline"
              onClick={addLayer}
              type="button"
              style={{ height: '28px', padding: '0 12px', fontSize: '11px' }}
            >
              + ADD
            </button>
          </div>
        </div>
        
        <div className={styles.layers}>
          {filmLayers.map((layer, index) => (
            <div key={index} className="layer-item">
              <div className={`layer-item__header ${styles.layerHeader} ${expandedLayers.has(index) ? '' : styles.layerHeaderCollapsed}`}>
                <div className={styles.layerHeaderLeft}>
                  <button
                    className={styles.collapseBtn}
                    type="button"
                    onClick={() => toggleLayer(index)}
                    aria-expanded={expandedLayers.has(index)}
                    title={expandedLayers.has(index) ? 'Collapse layer' : 'Expand layer'}
                  >
                    {expandedLayers.has(index) ? '▼' : '▶'}
                  </button>
                  <input
                    type="text"
                    value={layer.name}
                    onChange={(e) => updateLayer(index, 'name', e.target.value)}
                    className={styles.nameInput}
                  />
                </div>
                {index !== 0 && index !== filmLayers.length - 1 && (
                  <button 
                    className="layer-item__remove" 
                    onClick={() => removeLayer(index)}
                    type="button"
                  >
                    ×
                  </button>
                )}
              </div>

              {expandedLayers.has(index) && (
                <div className={styles.layerParams}>
                <div className="control">
                  <div className="control__label">
                    <span>SLD</span>
                    <EditableValue
                      value={layer.sld}
                      onChange={(v) => updateLayer(index, 'sld', v)}
                      min={0}
                      max={10}
                      step={0.01}
                      decimals={2}
                    />
                  </div>
                  <input
                    type="range"
                    className="slider"
                    min="0"
                    max="6.4"
                    step="0.01"
                    value={layer.sld}
                    onChange={(e) => updateLayer(index, 'sld', parseFloat(e.target.value))}
                  />
                </div>

                <div className="control">
                  <div className="control__label">
                    <span>Thickness (Å)</span>
                    <EditableValue
                      value={layer.thickness}
                      onChange={(v) => updateLayer(index, 'thickness', v)}
                      min={0}
                      max={2000}
                      step={1}
                      decimals={0}
                      disabled={index === 0 || index === filmLayers.length - 1}
                    />
                  </div>
                  <input
                    type="range"
                    className="slider"
                    min="0"
                    max="500"
                    step="1"
                    value={layer.thickness}
                    onChange={(e) => updateLayer(index, 'thickness', parseFloat(e.target.value))}
                    disabled={index === 0 || index === filmLayers.length - 1}
                  />
                </div>

                <div className="control">
                  <div className="control__label">
                    <span>Roughness (Å)</span>
                    <EditableValue
                      value={layer.roughness}
                      onChange={(v) => updateLayer(index, 'roughness', v)}
                      min={0}
                      max={500}
                      step={0.5}
                      decimals={1}
                    />
                  </div>
                  <input
                    type="range"
                    className="slider"
                    min="0"
                    max="150"
                    step="0.5"
                    value={layer.roughness}
                    onChange={(e) => updateLayer(index, 'roughness', parseFloat(e.target.value))}
                  />
                </div>

                <div className="control">
                  <div className="control__label">
                    <span>iSLD</span>
                    <EditableValue
                      value={layer.isld}
                      onChange={(v) => updateLayer(index, 'isld', v)}
                      min={0}
                      max={1}
                      step={0.001}
                      decimals={3}
                    />
                  </div>
                  <input
                    type="range"
                    className="slider"
                    min="0"
                    max="1"
                    step="0.001"
                    value={layer.isld}
                    onChange={(e) => updateLayer(index, 'isld', parseFloat(e.target.value))}
                  />
                </div>
              </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Generator Parameters */}
      <div className="section">
        <div className="section__header">
          <h3 className="section__title">Generator</h3>
        </div>
        
        <div className="control">
          <div className="control__label">
            <span>Number of Curves{isProduction && ` (max ${limits.max_curves})`}</span>
            <EditableValue
              value={generatorParams.numCurves}
              onChange={(v) => onGeneratorParamsChange({ ...generatorParams, numCurves: Math.min(Math.round(v), limits.max_curves) })}
              min={10}
              max={limits.max_curves}
              step={10}
              decimals={0}
            />
          </div>
          <input
            type="range"
            className="slider"
            min="100"
            max={limits.max_curves}
            step="100"
            value={generatorParams.numCurves}
            onChange={(e) => onGeneratorParamsChange({ 
              ...generatorParams, 
              numCurves: parseInt(e.target.value) 
            })}
          />
        </div>

        <div className="control">
          <div className="control__label">
            <span>Max Film Layers</span>
            <EditableValue
              value={generatorParams.numFilmLayers}
              onChange={(v) => onGeneratorParamsChange({ ...generatorParams, numFilmLayers: Math.min(Math.max(1, Math.round(v)), filmLayers.length) })}
              min={1}
              max={filmLayers.length}
              step={1}
              decimals={0}
            />
          </div>
          <input
            type="range"
            className="slider"
            min="1"
            max={filmLayers.length}
            step="1"
            value={generatorParams.numFilmLayers}
            onChange={(e) => onGeneratorParamsChange({ 
              ...generatorParams, 
              numFilmLayers: Math.min(parseInt(e.target.value), filmLayers.length) 
            })}
          />
        </div>
      </div>

      {/* Training Parameters */}
      <div className="section">
        <div className="section__header">
          <h3 className="section__title">Training</h3>
        </div>
        
        <div className="control">
          <div className="control__label">
            <span>Batch Size{isProduction && ` (max ${limits.max_batch_size})`}</span>
            <EditableValue
              value={trainingParams.batchSize}
              onChange={(v) => onTrainingParamsChange({ ...trainingParams, batchSize: Math.min(Math.round(v), limits.max_batch_size) })}
              min={1}
              max={limits.max_batch_size}
              step={8}
              decimals={0}
            />
          </div>
          <input
            type="range"
            className="slider"
            min="8"
            max={limits.max_batch_size}
            step="8"
            value={trainingParams.batchSize}
            onChange={(e) => onTrainingParamsChange({ 
              ...trainingParams, 
              batchSize: parseInt(e.target.value) 
            })}
          />
        </div>

        <div className="control">
          <div className="control__label">
            <span>Epochs{isProduction && ` (max ${limits.max_epochs})`}</span>
            <EditableValue
              value={trainingParams.epochs}
              onChange={(v) => onTrainingParamsChange({ ...trainingParams, epochs: Math.min(Math.round(v), limits.max_epochs) })}
              min={1}
              max={limits.max_epochs}
              step={1}
              decimals={0}
            />
          </div>
          <input
            type="range"
            className="slider"
            min="1"
            max={limits.max_epochs}
            step="1"
            value={trainingParams.epochs}
            onChange={(e) => onTrainingParamsChange({ 
              ...trainingParams, 
              epochs: parseInt(e.target.value) 
            })}
          />
        </div>

        <div className="control">
          <div className="control__label">
            <span>CNN Layers{isProduction && ` (max ${limits.max_cnn_layers})`}</span>
            <EditableValue
              value={trainingParams.layers}
              onChange={(v) => onTrainingParamsChange({ ...trainingParams, layers: Math.min(Math.round(v), limits.max_cnn_layers) })}
              min={1}
              max={limits.max_cnn_layers}
              step={1}
              decimals={0}
            />
          </div>
          <input
            type="range"
            className="slider"
            min="1"
            max={limits.max_cnn_layers}
            step="1"
            value={trainingParams.layers}
            onChange={(e) => onTrainingParamsChange({ 
              ...trainingParams, 
              layers: parseInt(e.target.value) 
            })}
          />
        </div>

        <div className="control">
          <div className="control__label">
            <span>Dropout{isProduction && ` (max ${limits.max_dropout})`}</span>
            <EditableValue
              value={trainingParams.dropout}
              onChange={(v) => onTrainingParamsChange({ ...trainingParams, dropout: Math.min(v, limits.max_dropout) })}
              min={0}
              max={limits.max_dropout}
              step={0.01}
              decimals={2}
            />
          </div>
          <input
            type="range"
            className="slider"
            min="0"
            max={limits.max_dropout}
            step="0.05"
            value={trainingParams.dropout}
            onChange={(e) => onTrainingParamsChange({ 
              ...trainingParams, 
              dropout: parseFloat(e.target.value) 
            })}
          />
        </div>

        <div className="control">
          <div className="control__label">
            <span>Latent Dimension{isProduction && ` (max ${limits.max_latent_dim})`}</span>
            <EditableValue
              value={trainingParams.latentDim}
              onChange={(v) => onTrainingParamsChange({ ...trainingParams, latentDim: Math.min(Math.round(v), limits.max_latent_dim) })}
              min={2}
              max={limits.max_latent_dim}
              step={4}
              decimals={0}
            />
          </div>
          <input
            type="range"
            className="slider"
            min="4"
            max={limits.max_latent_dim}
            step="4"
            value={trainingParams.latentDim}
            onChange={(e) => onTrainingParamsChange({ 
              ...trainingParams, 
              latentDim: parseInt(e.target.value) 
            })}
          />
        </div>

        <div className="control">
          <div className="control__label">
            <span>AE Epochs{isProduction && ` (max ${limits.max_ae_epochs})`}</span>
            <EditableValue
              value={trainingParams.aeEpochs}
              onChange={(v) => onTrainingParamsChange({ ...trainingParams, aeEpochs: Math.min(Math.round(v), limits.max_ae_epochs) })}
              min={1}
              max={limits.max_ae_epochs}
              step={10}
              decimals={0}
            />
          </div>
          <input
            type="range"
            className="slider"
            min="10"
            max={limits.max_ae_epochs}
            step="10"
            value={trainingParams.aeEpochs}
            onChange={(e) => onTrainingParamsChange({ 
              ...trainingParams, 
              aeEpochs: parseInt(e.target.value) 
            })}
          />
        </div>

        <div className="control">
          <div className="control__label">
            <span>MLP Epochs{isProduction && ` (max ${limits.max_mlp_epochs})`}</span>
            <EditableValue
              value={trainingParams.mlpEpochs}
              onChange={(v) => onTrainingParamsChange({ ...trainingParams, mlpEpochs: Math.min(Math.round(v), limits.max_mlp_epochs) })}
              min={1}
              max={limits.max_mlp_epochs}
              step={10}
              decimals={0}
            />
          </div>
          <input
            type="range"
            className="slider"
            min="10"
            max={limits.max_mlp_epochs}
            step="10"
            value={trainingParams.mlpEpochs}
            onChange={(e) => onTrainingParamsChange({ 
              ...trainingParams, 
              mlpEpochs: parseInt(e.target.value) 
            })}
          />
        </div>
      </div>

      {/* Data Upload */}
      <div className="section">
        <div className="section__header">
          <h3 className="section__title">Data & Models</h3>
        </div>

        <div className={styles.uploadArea}>
          <div
            className={`${styles.uploadDropzone} ${isDragActive ? styles.uploadDropzoneActive : ''}`}
            onClick={handleDropzoneClick}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <span className={styles.uploadIcon}>↑</span>
            <span className={styles.uploadText}>
              <span className={styles.uploadTextBold}>Click</span> or drag files
            </span>
            <span className={styles.uploadText}>.npy, .pth, .pt</span>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".npy,.pth,.pt,.yml,.yaml"
            className={styles.fileInputHidden}
            onChange={handleFileChange}
            disabled={isUploading}
          />

          {selectedFiles.length > 0 && (
            <div className={styles.selectedFiles}>
              {selectedFiles.map((file, index) => (
                <div key={index} className={styles.selectedFile}>
                  <span className={styles.selectedFileName}>{file.name}</span>
                  <button
                    className={styles.selectedFileRemove}
                    onClick={() => removeSelectedFile(index)}
                    type="button"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}

          {selectedFiles.length > 0 && (
            <div className={styles.uploadActions}>
              <button
                className="btn btn--outline btn--full"
                onClick={handleUpload}
                disabled={isUploading}
              >
                {isUploading ? 'UPLOADING...' : `UPLOAD ${selectedFiles.length} FILE${selectedFiles.length > 1 ? 'S' : ''}`}
              </button>
            </div>
          )}

          <div className={styles.uploadHint}>
            Upload datasets & pretrained models
          </div>

          {backendStatus && (backendStatus.data_files.length > 0 || backendStatus.curve_files.length > 0) && (
            <div className={styles.availableFiles}>
              <div className={styles.availableFilesLabel}>Available:</div>
              {backendStatus.data_files.map((f) => (
                <span key={f} className={styles.availableFile}>{f}</span>
              ))}
              {backendStatus.curve_files.map((f) => (
                <span key={f} className={styles.availableFile}>{f}</span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="section">
        <div className={styles.actionRow}>
          <button
            className="btn btn--outline"
            onClick={onReset}
            type="button"
            disabled={isGenerating || isUploading}
          >
            RESET
          </button>
          <button
            className="btn btn--full"
            onClick={onGenerate}
            disabled={isGenerating}
          >
            {isGenerating ? 'GENERATING...' : 'GENERATE'}
          </button>
        </div>
      </div>
    </div>
  );
}
