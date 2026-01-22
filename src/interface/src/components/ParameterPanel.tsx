'use client';

import { useMemo, useState, useRef, useCallback, useEffect, type ChangeEvent } from 'react';
import {
  FilmLayer,
  GeneratorParams,
  TrainingParams,
  Limits,
  DEFAULT_LIMITS,
  DataSource,
  Workflow,
  NrSldMode,
  UploadRole,
  GpuTier,
  GPU_TIERS,
  type LayerBound,
  type LayerBoundParam,
} from '@/types';
import EditableValue from './EditableValue';
import InfoTooltip from './InfoTooltip';
import RangeSlider from './RangeSlider';
import styles from './ParameterPanel.module.css';

interface BackendStatus {
  pyreflect_available: boolean;
  has_settings: boolean;
  data_files: string[];
  curve_files: string[];
  expt_files: string[];
  model_files?: string[];
  settings_paths?: {
    nr_predict_sld?: {
      file?: Record<string, string>;
      models?: {
        model?: string;
        normalization_stats?: string;
      };
    };
    sld_predict_chi?: {
      file?: Record<string, string>;
    };
  };
  settings_status?: {
    nr_predict_sld?: {
      file?: Record<string, boolean>;
      models?: Record<string, boolean>;
    };
    sld_predict_chi?: {
      file?: Record<string, boolean>;
    };
  };
}

export interface ParameterPanelProps {
  filmLayers: FilmLayer[];
  generatorParams: GeneratorParams;
  trainingParams: TrainingParams;
  onFilmLayersChange: (layers: FilmLayer[]) => void;
  onGeneratorParamsChange: (params: GeneratorParams) => void;
  onTrainingParamsChange: (params: TrainingParams) => void;
  onGenerate: (name?: string) => void;
  onReset: () => void;
  onUploadFiles: (files: { file: File; role: UploadRole }[]) => void;
  isGenerating: boolean;
  isUploading: boolean;
  backendStatus: BackendStatus | null;
  dataSource: DataSource;
  workflow: Workflow;
  nrSldMode: NrSldMode;
  autoGenerateModelStats: boolean;
  onDataSourceChange: (value: DataSource) => void;
  onWorkflowChange: (value: Workflow) => void;
  onNrSldModeChange: (value: NrSldMode) => void;
  onAutoGenerateModelStatsChange: (value: boolean) => void;
  limits?: Limits;
  isProduction?: boolean;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
  gpu?: GpuTier;
  onGpuChange?: (value: GpuTier) => void;
}

const DEFAULT_LAYERS: FilmLayer[] = [
  { name: 'substrate', sld: 2.07, isld: 0, thickness: 0, roughness: 1.35 },
  { name: 'siox', sld: 3.47, isld: 0, thickness: 12.17, roughness: 2.05 },
  { name: 'layer_1', sld: 3.96, isld: 0, thickness: 53.79, roughness: 20.61 },
  { name: 'layer_2', sld: 2.37, isld: 0, thickness: 178.66, roughness: 57.26 },
  { name: 'layer_3', sld: 3.85, isld: 0, thickness: 79.63, roughness: 21.86 },
  { name: 'layer_4', sld: 3.24, isld: 0, thickness: 67.49, roughness: 17.57 },
  { name: 'layer_5', sld: 2.67, isld: 0, thickness: 72.61, roughness: 76.63 },
  { name: 'air', sld: 0, isld: 0, thickness: 0, roughness: 0 },
];

const DEFAULT_BOUNDS_LIST: LayerBound[] = [
    // substrate
    { i: 0, par: 'roughness', bounds: [1.177, 1.5215] },
    // silicon oxide layer
    { i: 1, par: 'sld', bounds: [3.47, 3.47] },
    { i: 1, par: 'thickness', bounds: [9.7216, 14.624] },
    { i: 1, par: 'roughness', bounds: [1.108, 2.998] },
    // material layer 1
    { i: 2, par: 'sld', bounds: [3.7235, 4.197] },
    { i: 2, par: 'thickness', bounds: [8.717, 98.867] },
    { i: 2, par: 'roughness', bounds: [2.2571, 38.969] },
    // material layer 2
    { i: 3, par: 'sld', bounds: [1.6417, 3.1033] },
    { i: 3, par: 'thickness', bounds: [117.4, 239.91] },
    { i: 3, par: 'roughness', bounds: [19.32, 95.202] },
    // material layer 3
    { i: 4, par: 'sld', bounds: [3.0246, 4.6755] },
    { i: 4, par: 'thickness', bounds: [64.482, 94.768] },
    { i: 4, par: 'roughness', bounds: [15.713, 28.007] },
    // material layer 4
    { i: 5, par: 'sld', bounds: [1.501, 4.9837] },
    { i: 5, par: 'thickness', bounds: [51.655, 83.334] },
    { i: 5, par: 'roughness', bounds: [9.7741, 25.373] },
    // material layer 5
    { i: 6, par: 'sld', bounds: [0.85516, 4.4906] },
    { i: 6, par: 'thickness', bounds: [58.479, 86.738] },
    { i: 6, par: 'roughness', bounds: [43.155, 110.11] },
];

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
  dataSource,
  workflow,
  nrSldMode,
  autoGenerateModelStats,
  onDataSourceChange,
  onWorkflowChange,
  onNrSldModeChange,
  onAutoGenerateModelStatsChange,
  limits = DEFAULT_LIMITS,
  isProduction = false,
  isCollapsed = false,
  onToggleCollapse,
  gpu = 'T4',
  onGpuChange,
}: ParameterPanelProps) {
  const [expandedLayers, setExpandedLayers] = useState(() =>
    new Set<number>(filmLayers.map((_, index) => index))
  );

  const [showNamePopup, setShowNamePopup] = useState(false);
  const [generationName, setGenerationName] = useState('');

  // Track independent bounds per layer/parameter
  // Key: "layerIndex:param", Value: { min: number | null, max: number | null }
  const [layerBounds, setLayerBounds] = useState<Record<string, { min: number | null; max: number | null }>>({});

  const derivedNumFilmLayers = Math.max(0, filmLayers.length - 3);

  // Get bounds for a specific layer/param
  const getBounds = useCallback((layerIndex: number, param: LayerBoundParam): { min: number | null; max: number | null } => {
    return layerBounds[`${layerIndex}:${param}`] || { min: null, max: null };
  }, [layerBounds]);

  // Set bounds for a specific layer/param
  const setBounds = useCallback((layerIndex: number, param: LayerBoundParam, min: number | null, max: number | null) => {
    setLayerBounds((prev) => ({
      ...prev,
      [`${layerIndex}:${param}`]: { min, max },
    }));
  }, []);

  // Check if any bounds are set (bounds mode active)
  const hasAnyBounds = useMemo(() => {
    return Object.values(layerBounds).some((b) => b.min !== null || b.max !== null);
  }, [layerBounds]);

  // Sync incoming generatorParams.layerBound into local layerBounds state on mount/change
  // This ensures that loading from saved state (which has bounds) correctly populates the UI
  useEffect(() => {
    // Only sync if local state is empty but prop has data (initial load),
    // OR if we suspect a hard external reset happened that didn't go through resetAllLayers.
    // However, syncing complex state bi-directionally is risky (infinite loops).
    // Safest approach: Init state lazy from prop if possible, or use a ref to track if we've initialized.
    // But since we can't change useState init easily without key-remounting, let's use an Effect that runs once or when prop significantly changes?
    // Actually, simply parsing the prop into the map structure if the map is empty is a good start.
    
    if (Object.keys(layerBounds).length === 0 && generatorParams.layerBound && generatorParams.layerBound.length > 0) {
        const newBounds: Record<string, { min: number | null; max: number | null }> = {};
        generatorParams.layerBound.forEach((b) => {
            newBounds[`${b.i}:${b.par}`] = {
                min: b.bounds[0],
                max: b.bounds[1]
            };
        });
        setLayerBounds(newBounds);
    }
  }, [generatorParams.layerBound]); // Intentionally do NOT include layerBounds in dep array to avoid loops, or check emptiness guard.

  // Build layerBound array from bounds for backend
  const buildLayerBoundsPayload = useCallback((): LayerBound[] | undefined => {
    const bounds: LayerBound[] = [];
    const pars: LayerBoundParam[] = ['sld', 'isld', 'thickness', 'roughness'];

    for (let i = 0; i < filmLayers.length; i++) {
      const layer = filmLayers[i];
      for (const par of pars) {
        const b = layerBounds[`${i}:${par}`];
        if (b && (b.min !== null || b.max !== null)) {
          const value = layer[par] || 0;
          // Use value as fallback for missing bound
          const boundMin = b.min ?? value;
          const boundMax = b.max ?? value;
          bounds.push({
            i,
            par,
            bounds: [boundMin, boundMax],
          });
        }
      }
    }

    return bounds.length > 0 ? bounds : undefined;
  }, [filmLayers, layerBounds]);

  // Sync layerBound to generatorParams whenever bounds change
  useEffect(() => {
    const newBounds = buildLayerBoundsPayload();
    const currentBounds = generatorParams.layerBound;

    // Only update if bounds actually changed
    const newJson = JSON.stringify(newBounds || null);
    const currentJson = JSON.stringify(currentBounds || null);

    if (newJson !== currentJson) {
      onGeneratorParamsChange({
        ...generatorParams,
        numFilmLayers: hasAnyBounds ? derivedNumFilmLayers : generatorParams.numFilmLayers,
        layerBound: newBounds,
      });
    }
  }, [
    layerBounds,
    filmLayers,
    buildLayerBoundsPayload,
    generatorParams,
    onGeneratorParamsChange,
    hasAnyBounds,
    derivedNumFilmLayers,
  ]);

  // Clear all bounds
  const clearAllBounds = useCallback(() => {
    setLayerBounds({});
  }, []);

  // Get default value for a layer index/field
  const getLayerDefault = (index: number, field: keyof FilmLayer): number | string => {
    if (index >= 0 && index < DEFAULT_LAYERS.length) {
      return DEFAULT_LAYERS[index][field];
    }
    // Generic defaults for new user-added layers
    if (field === 'sld') return 3.0;
    if (field === 'thickness') return 100;
    if (field === 'roughness') return 20;
    if (field === 'isld') return 0;
    if (field === 'name') return 'layer';
    return 0;
  };

  // Get default bounds for a layer index/parameter
  const getBoundDefault = (index: number, param: LayerBoundParam): { min: number | null, max: number | null } => {
    // Look for exact match in DEFAULT_BOUNDS_LIST
    const defaultBound = DEFAULT_BOUNDS_LIST.find(b => b.i === index && b.par === param);
    if (defaultBound) {
      return { min: defaultBound.bounds[0], max: defaultBound.bounds[1] };
    }
    return { min: null, max: null };
  };

  const resetAllLayers = () => {
    // Reset each layer to its matching default from DEFAULT_LAYERS
    // If exact name match not found (e.g. layer_X vs layer_Y), fallback to hard reset,
    // but better to try to map by index if count matches, or name if unique.
    // Given the task, let's just restore the structure entirely to DEFAULT_LAYERS?
    // User requested "reset all layers to default values".
    // If they added extra layers, "reset" usually means "reset values of current layers" OR "restore original state"?
    
    // Interpretation: "Reset values" of CURRENT layers.
    // But defaults are specific! (e.g. siox sld=3.47).
    // Let's restore the full DEFAULT_LAYERS stack which ensures all values are correct.
    // If the user ADDED layers, "Resetting to default" usually implies going back to the start state.
    // However, if they want to keep their custom layer structure but just reset numbers...
    // The previous implementation was: sld: 3.0, thickness: 100... generic defaults.
    // But the "Overwrite" button does exactly what loadDefaultLayers does.
    // Let's make "Reset" do the same thing as "Overwrite" to be consistent?
    // Or should "Reset" keep the layers but reset their values?
    // "overwrting and resetting give different values" implies they SHOULD give the same values (or at least consistent ones).
    // Let's make resetAllLayers simply call loadDefaultLayers() to be safe and consistent.
    
    loadDefaultLayers();
  };

  const loadDefaultLayers = () => {
    onFilmLayersChange(DEFAULT_LAYERS);
    
    // Construct bounds dictionary from list
    const newBounds: Record<string, { min: number | null; max: number | null }> = {};
    DEFAULT_BOUNDS_LIST.forEach((item) => {
        newBounds[`${item.i}:${item.par}`] = {
            min: item.bounds[0],
            max: item.bounds[1]
        };
    });
    setLayerBounds(newBounds);
    
    // Expand all
    setExpandedLayers(new Set(DEFAULT_LAYERS.map((_, index) => index)));
  };

  const shouldShowTraining =
    dataSource === 'synthetic' ||
    (dataSource === 'real' && (workflow === 'sld_chi' || workflow === 'nr_sld_chi' || nrSldMode === 'train'));

  const trainingTooltip =
    dataSource === 'synthetic'
      ? 'Used to train the synthetic generator/film stack for this run.'
      : workflow === 'nr_sld' && nrSldMode === 'train'
        ? 'Needed to train the NR→SLD model from nr_train/sld_train. Not used in infer.'
        : workflow === 'sld_chi'
          ? 'Used to train the SLD→Chi model from SLD/chi training files.'
          : workflow === 'nr_sld_chi' && nrSldMode === 'train'
            ? 'Applies to the training stages in this pipeline: NR→SLD training (and SLD→Chi training if enabled).'
            : 'Training controls are only used when the pipeline includes a training stage.';

  const settingsPaths = backendStatus?.settings_paths;
  const settingsStatus = backendStatus?.settings_status;
  const stripKnownExtension = (path: string): string =>
    path.replace(/\.(npy|npz|json|pth|pt|yml|yaml)$/i, '');
  const stripLabelSuffix = (label: string): string => label.replace(/\s*\([^)]*\)\s*$/, '');
  const getSettingPath = (role: UploadRole): string | undefined => {
    if (!settingsPaths) return undefined;
    switch (role) {
      case 'nr_train':
        return settingsPaths.nr_predict_sld?.file?.nr_train;
      case 'sld_train':
        return settingsPaths.nr_predict_sld?.file?.sld_train;
      case 'experimental_nr':
        return settingsPaths.nr_predict_sld?.file?.experimental_nr_file;
      case 'normalization_stats':
        return settingsPaths.nr_predict_sld?.models?.normalization_stats;
      case 'nr_sld_model':
        return settingsPaths.nr_predict_sld?.models?.model;
      case 'sld_chi_experimental_profile':
        return settingsPaths.sld_predict_chi?.file?.model_experimental_sld_profile;
      case 'sld_chi_model_sld_file':
        return settingsPaths.sld_predict_chi?.file?.model_sld_file;
      case 'sld_chi_model_chi_params_file':
        return settingsPaths.sld_predict_chi?.file?.model_chi_params_file;
      default:
        return undefined;
    }
  };

  const hasSettingFile = (role: UploadRole): boolean => {
    if (!settingsStatus) return false;
    switch (role) {
      case 'nr_train':
        return !!settingsStatus.nr_predict_sld?.file?.nr_train;
      case 'sld_train':
        return !!settingsStatus.nr_predict_sld?.file?.sld_train;
      case 'experimental_nr':
        return !!settingsStatus.nr_predict_sld?.file?.experimental_nr_file;
      case 'normalization_stats':
        return !!settingsStatus.nr_predict_sld?.models?.normalization_stats;
      case 'nr_sld_model':
        return !!settingsStatus.nr_predict_sld?.models?.model;
      case 'sld_chi_experimental_profile':
        return !!settingsStatus.sld_predict_chi?.file?.model_experimental_sld_profile;
      case 'sld_chi_model_sld_file':
        return !!settingsStatus.sld_predict_chi?.file?.model_sld_file;
      case 'sld_chi_model_chi_params_file':
        return !!settingsStatus.sld_predict_chi?.file?.model_chi_params_file;
      default:
        return false;
    }
  };

  const uploadRequirementLabels: Record<UploadRole, string> = {
    auto: 'Auto',
    nr_train: 'NR Train (.npy)',
    sld_train: 'SLD Train (.npy)',
    experimental_nr: 'Experimental NR (.npy)',
    normalization_stats: 'Normalization Stats (.npz/.json)',
    nr_sld_model: 'NR→SLD Model (.pth/.pt)',
    sld_chi_experimental_profile: 'SLD→Chi Experimental (.npy)',
    sld_chi_model_sld_file: 'SLD→Chi SLD Train (.npy)',
    sld_chi_model_chi_params_file: 'SLD→Chi Chi Params (.npy)',
  };

  const requiredUploads = (() => {
    if (dataSource !== 'real') return [];
    if (workflow === 'sld_chi') {
      return ['sld_chi_experimental_profile', 'sld_chi_model_sld_file', 'sld_chi_model_chi_params_file'] as const;
    }
    if (workflow === 'nr_sld_chi') {
      if (nrSldMode === 'infer') {
        return ['experimental_nr', 'nr_sld_model', 'normalization_stats', 'sld_chi_model_sld_file', 'sld_chi_model_chi_params_file'];
      }
      return [
        'nr_train',
        'sld_train',
        ...(autoGenerateModelStats ? [] : ['nr_sld_model', 'normalization_stats']),
        'sld_chi_model_sld_file',
        'sld_chi_model_chi_params_file',
      ] as const;
    }
    if (nrSldMode === 'infer') {
      return ['experimental_nr', 'nr_sld_model', 'normalization_stats'] as const;
    }
    return ['nr_train', 'sld_train', ...(autoGenerateModelStats ? [] : ['nr_sld_model', 'normalization_stats'])] as const;
  })() as UploadRole[];

  const pipelineRoles = (() => {
    if (dataSource !== 'real') return [];
    if (workflow === 'sld_chi') {
      return ['sld_chi_experimental_profile', 'sld_chi_model_sld_file', 'sld_chi_model_chi_params_file'] as const;
    }
    if (workflow === 'nr_sld_chi') {
      if (nrSldMode === 'infer') {
        return [
          'experimental_nr',
          'nr_sld_model',
          'normalization_stats',
          'sld_chi_model_sld_file',
          'sld_chi_model_chi_params_file',
        ] as const;
      }
      return [
        'nr_train',
        'sld_train',
        'nr_sld_model',
        'normalization_stats',
        'sld_chi_model_sld_file',
        'sld_chi_model_chi_params_file',
      ] as const;
    }
    if (nrSldMode === 'infer') {
      return ['experimental_nr', 'nr_sld_model', 'normalization_stats'] as const;
    }
    return ['nr_train', 'sld_train', 'nr_sld_model', 'normalization_stats'] as const;
  })() as UploadRole[];

  const missingUploads = requiredUploads.filter((role) => !hasSettingFile(role));
  const canGenerate = dataSource === 'synthetic' || (requiredUploads.length === 0 ? true : missingUploads.length === 0);
  const generateBlockReason = missingUploads.length
    ? `Missing: ${missingUploads.map((role) => uploadRequirementLabels[role]).join(', ')}`
    : undefined;

  const uploadInputRef = useRef<HTMLInputElement>(null);
  const [pendingUploadRole, setPendingUploadRole] = useState<UploadRole | null>(null);

  const roleAcceptMap: Record<UploadRole, string> = {
    auto: '.npy,.pth,.pt',
    nr_train: '.npy',
    sld_train: '.npy',
    experimental_nr: '.npy',
    normalization_stats: '.npz,.json',
    nr_sld_model: '.pth,.pt',
    sld_chi_experimental_profile: '.npy',
    sld_chi_model_sld_file: '.npy',
    sld_chi_model_chi_params_file: '.npy',
  };

  const handleRoleUploadClick = (role: UploadRole) => {
    setPendingUploadRole(role);
    uploadInputRef.current?.click();
  };

  const handleRoleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !pendingUploadRole) return;
    onUploadFiles([{ file, role: pendingUploadRole }]);
    setPendingUploadRole(null);
    event.target.value = '';
  };

  const handleGenerateClick = () => {
    setShowNamePopup(true);
  };

  const handleStartGeneration = () => {
    setShowNamePopup(false);
    onGenerate(generationName || undefined);
    setGenerationName('');
  };

  const handleCancelGeneration = () => {
    setShowNamePopup(false);
    setGenerationName('');
  };

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
    // Find the next layer number by looking at existing layer names
    const existingNumbers = filmLayers
      .map((l) => {
        const match = l.name.match(/^layer_(\d+)$/);
        return match ? parseInt(match[1], 10) : 0;
      })
      .filter((n) => n > 0);
    const nextNumber = existingNumbers.length > 0 ? Math.max(...existingNumbers) + 1 : 1;

    const newLayer: FilmLayer = {
      name: `layer_${nextNumber}`,
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

  const resetLayer = (index: number) => {
    const newLayers = [...filmLayers];
    // If we have a default for this index, use it fully
    if (index < DEFAULT_LAYERS.length) {
      newLayers[index] = { ...DEFAULT_LAYERS[index] };
    } else {
      // Keep name, reset others to generic
      newLayers[index] = {
        ...newLayers[index],
        sld: 3.0,
        thickness: 100,
        roughness: 20,
        isld: 0,
      };
    }
    onFilmLayersChange(newLayers);
    
    // Reset bounds for this layer to their defaults (if any) or clear them
    setLayerBounds((prev) => {
      const next = { ...prev };
      // First clear existing for this layer
      Object.keys(next).forEach((key) => {
        if (key.startsWith(`${index}:`)) {
          delete next[key];
        }
      });
      // Then re-apply defaults if any
      const pars: LayerBoundParam[] = ['sld', 'isld', 'thickness', 'roughness'];
      for (const par of pars) {
        const def = getBoundDefault(index, par);
        if (def.min !== null || def.max !== null) {
          next[`${index}:${par}`] = def;
        }
      }
      return next;
    });
  };

  if (isCollapsed) {
    return (
      <div className={styles.panelCollapsed}>
        <button
          className={styles.sidebarToggle}
          onClick={onToggleCollapse}
          title="Expand Sidebar"
        >
          ▶
        </button>
      </div>
    );
  }

  return (
    <div className={styles.panel}>
      {/* Workflow */}
      <div className="section">
        <div className="section__header">
          <h3 className="section__title">Workflow</h3>
          {onToggleCollapse && (
            <div className={styles.sectionActions}>
              <button
                className={styles.sidebarToggle}
                onClick={onToggleCollapse}
                type="button"
                title="Collapse Sidebar"
              >
                ◀
              </button>
            </div>
          )}
        </div>

        <div className="control">
          <div className="control__label">
            <span>
              Data Source
              <InfoTooltip hint={"Synthetic uses film layers to generate data.\n\nReal data uses settings.yml files and ignores film layers and generator settings."} />
            </span>
          </div>
          <select
            className="control__input"
            value={dataSource}
            onChange={(e) => onDataSourceChange(e.target.value as DataSource)}
          >
            <option value="synthetic" title="Generate synthetic curves from film layers.">
              Synthetic (film layers)
            </option>
            <option value="real" title="Use uploaded .npy files referenced in settings.yml.">
              Real data (.npy)
            </option>
          </select>
        </div>

        {dataSource === 'real' && (
          <>
            <div className="control">
              <div className="control__label">
                <span>
                  Pipeline
                  <InfoTooltip hint={"NR → SLD predicts SLD from NR.\n\nSLD → Chi predicts chi from SLD.\n\nNR → SLD → Chi chains both in one run and uses predicted SLD as chi input."} />
                </span>
              </div>
              <select
                className="control__input"
                value={workflow}
                onChange={(e) => onWorkflowChange(e.target.value as Workflow)}
              >
                <option value="nr_sld" title="Predict SLD profiles from NR curves.">
                  NR → SLD
                </option>
                <option value="sld_chi" title="Predict chi parameters from SLD profiles.">
                  SLD → Chi
                </option>
                <option value="nr_sld_chi" title="Predict SLD from NR, then chi from predicted SLD.">
                  NR → SLD → Chi
                </option>
              </select>
            </div>

            {(workflow === 'nr_sld' || workflow === 'nr_sld_chi') && (
              <div className="control">
                <div className="control__label">
                  <span>
                    Mode
                    <InfoTooltip hint={"Train: uses nr_train/sld_train to train NR→SLD and (if enabled) generate model + stats. In NR→SLD→Chi, chi is predicted from the trained SLD output.\n\nInfer: uses experimental_nr with existing model + stats to predict SLD. In NR→SLD→Chi, chi is predicted from that inferred SLD."} />
                  </span>
                </div>
                <select
                  className="control__input"
                  value={nrSldMode}
                  onChange={(e) => onNrSldModeChange(e.target.value as NrSldMode)}
                >
                  <option value="train" title="Train a new NR→SLD model from train datasets.">
                    Train
                  </option>
                  <option value="infer" title="Run inference using experimental NR with existing model + stats.">
                    Infer
                  </option>
                </select>
              </div>
            )}

          </>
        )}
      </div>

      {dataSource === 'synthetic' && (
        <>
          {/* Film Layers Section */}
          <div className="section">
            <div className="section__header">
              <h3 className="section__title">
                Film Layers
                <InfoTooltip hint={"Film layers define a synthetic stack (materials and thickness) used for training data generation. Use the slider handles to set minimum and maximum bounds for each layer parameter."} />
              </h3>
              <div className={styles.layerActions}>
                <InfoTooltip hint="Reset all layers to default values">
                  <button
                    className={styles.clearBoundsBtn}
                    onClick={resetAllLayers}
                    type="button"
                    style={{ width: '28px', height: '28px' }}
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M18 6L6 18" />
                      <path d="M6 6l12 12" />
                    </svg>
                  </button>
                </InfoTooltip>
                <InfoTooltip hint="Overwrite with default layers.">
                <button
                  className="btn btn--outline"
                  onClick={loadDefaultLayers}
                  type="button"
                  style={{ height: '28px', padding: '0 8px', fontSize: '11px', display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                      <path d="M3 3v5h5" />
                  </svg>
                  
                </button>
                </InfoTooltip>
                <button
                  className="btn btn--outline"
                  onClick={expandedLayers.size ? collapseAllLayers : expandAllLayers}
                  type="button"
                  style={{ height: '28px', padding: '0 8px', fontSize: '11px' }}
                >
                  {expandedLayers.size ? 'COLLAPSE' : 'EXPAND'}
                </button>
                <button
                  className="btn btn--outline"
                  onClick={addLayer}
                  type="button"
                  style={{ height: '28px', padding: '0 8px', fontSize: '11px' }}
                >
                  ADD
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
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <InfoTooltip hint="Reset layer to defaults">
                        <button
                          className={styles.resetBtn}
                          onClick={(e) => {
                            e.stopPropagation();
                            resetLayer(index);
                          }}
                          type="button"
                          style={{ opacity: 1 }}
                        >
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                            <path d="M3 3v5h5" />
                          </svg>
                        </button>
                      </InfoTooltip>
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
                  </div>

                  {expandedLayers.has(index) && (
                    <div className={styles.layerParams}>
                      <div className="control">
                        <div className="control__label">
                          <span>SLD<InfoTooltip hint={"Scattering Length Density (×10⁻⁶ Å⁻²).\n\nDrag ◀ ▶ handles or edit [min] value [max] to set bounds."} /></span>
                          <div className={styles.valueWrapper}>
                            <InfoTooltip hint="Reset SLD to default">
                              <button
                                className={styles.resetBtn}
                                onClick={() => {
                                  updateLayer(index, 'sld', getLayerDefault(index, 'sld'));
                                  const def = getBoundDefault(index, 'sld');
                                  setBounds(index, 'sld', def.min, def.max);
                                }}
                              >
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                                  <path d="M3 3v5h5" />
                                </svg>
                              </button>
                            </InfoTooltip>
                            <div className={styles.valueWithBounds}>
                              {(getBounds(index, 'sld').min !== null || getBounds(index, 'sld').max !== null) ? (
                                <>
                                  <EditableValue
                                    value={getBounds(index, 'sld').min ?? layer.sld}
                                    onChange={(v) => setBounds(index, 'sld', v, getBounds(index, 'sld').max)}
                                    min={0}
                                    max={layer.sld}
                                    step={0.01}
                                    decimals={2}
                                  />
                                  <EditableValue
                                    value={getBounds(index, 'sld').max ?? layer.sld}
                                    onChange={(v) => setBounds(index, 'sld', getBounds(index, 'sld').min, v)}
                                    min={layer.sld}
                                    max={10}
                                    step={0.01}
                                    decimals={2}
                                  />
                                </>
                              ) : (
                                <EditableValue
                                  value={layer.sld}
                                  onChange={(v) => updateLayer(index, 'sld', v)}
                                  min={0}
                                  max={10}
                                  step={0.01}
                                  decimals={2}
                                />
                              )}
                            </div>
                          </div>
                        </div>
                        <RangeSlider
                          value={layer.sld}
                          boundMin={getBounds(index, 'sld').min}
                          boundMax={getBounds(index, 'sld').max}
                          min={0}
                          max={6.4}
                          step={0.01}
                          decimals={2}
                          onValueChange={(v) => updateLayer(index, 'sld', v)}
                          onBoundsChange={(lo, hi) => setBounds(index, 'sld', lo, hi)}
                        />
                      </div>

                      <div className="control">
                        <div className="control__label">
                          <span>Thickness (Å)<InfoTooltip hint={"Layer thickness in Angstroms.\n\nDrag ◀ ▶ handles or edit [min] value [max] to set bounds."} /></span>
                          <div className={styles.valueWrapper}>
                            <InfoTooltip hint="Reset Thickness to default">
                              <button
                                className={styles.resetBtn}
                                onClick={() => {
                                  updateLayer(index, 'thickness', getLayerDefault(index, 'thickness'));
                                  const def = getBoundDefault(index, 'thickness');
                                  setBounds(index, 'thickness', def.min, def.max);
                                }}
                                disabled={index === 0 || index === filmLayers.length - 1}
                              >
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                                  <path d="M3 3v5h5" />
                                </svg>
                              </button>
                            </InfoTooltip>
                            <div className={styles.valueWithBounds}>
                              {(getBounds(index, 'thickness').min !== null || getBounds(index, 'thickness').max !== null) ? (
                                <>
                                  <EditableValue
                                    value={getBounds(index, 'thickness').min ?? layer.thickness}
                                    onChange={(v) => setBounds(index, 'thickness', v, getBounds(index, 'thickness').max)}
                                    min={0}
                                    max={layer.thickness}
                                    step={1}
                                    decimals={0}
                                    disabled={index === 0 || index === filmLayers.length - 1}
                                  />
                                  <EditableValue
                                    value={getBounds(index, 'thickness').max ?? layer.thickness}
                                    onChange={(v) => setBounds(index, 'thickness', getBounds(index, 'thickness').min, v)}
                                    min={layer.thickness}
                                    max={2000}
                                    step={1}
                                    decimals={0}
                                    disabled={index === 0 || index === filmLayers.length - 1}
                                  />
                                </>
                              ) : (
                                <EditableValue
                                  value={layer.thickness}
                                  onChange={(v) => updateLayer(index, 'thickness', v)}
                                  min={0}
                                  max={2000}
                                  step={1}
                                  decimals={0}
                                  disabled={index === 0 || index === filmLayers.length - 1}
                                />
                              )}
                            </div>
                          </div>
                        </div>
                        <RangeSlider
                          value={layer.thickness}
                          boundMin={getBounds(index, 'thickness').min}
                          boundMax={getBounds(index, 'thickness').max}
                          min={0}
                          max={500}
                          step={1}
                          decimals={0}
                          onValueChange={(v) => updateLayer(index, 'thickness', v)}
                          onBoundsChange={(lo, hi) => setBounds(index, 'thickness', lo, hi)}
                          disabled={index === 0 || index === filmLayers.length - 1}
                        />
                      </div>

                      <div className="control">
                        <div className="control__label">
                          <span>Roughness (Å)<InfoTooltip hint={"Interfacial roughness in Angstroms.\n\nDrag ◀ ▶ handles or edit [min] value [max] to set bounds."} /></span>
                          <div className={styles.valueWrapper}>
                            <InfoTooltip hint="Reset Roughness to default">
                              <button
                                className={styles.resetBtn}
                                onClick={() => {
                                  updateLayer(index, 'roughness', getLayerDefault(index, 'roughness'));
                                  const def = getBoundDefault(index, 'roughness');
                                  setBounds(index, 'roughness', def.min, def.max);
                                }}
                              >
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                                  <path d="M3 3v5h5" />
                                </svg>
                              </button>
                            </InfoTooltip>
                            <div className={styles.valueWithBounds}>
                              {(getBounds(index, 'roughness').min !== null || getBounds(index, 'roughness').max !== null) ? (
                                <>
                                  <EditableValue
                                    value={getBounds(index, 'roughness').min ?? layer.roughness}
                                    onChange={(v) => setBounds(index, 'roughness', v, getBounds(index, 'roughness').max)}
                                    min={0}
                                    max={layer.roughness}
                                    step={0.5}
                                    decimals={1}
                                  />
                                  <EditableValue
                                    value={getBounds(index, 'roughness').max ?? layer.roughness}
                                    onChange={(v) => setBounds(index, 'roughness', getBounds(index, 'roughness').min, v)}
                                    min={layer.roughness}
                                    max={500}
                                    step={0.5}
                                    decimals={1}
                                  />
                                </>
                              ) : (
                                <EditableValue
                                  value={layer.roughness}
                                  onChange={(v) => updateLayer(index, 'roughness', v)}
                                  min={0}
                                  max={500}
                                  step={0.5}
                                  decimals={1}
                                />
                              )}
                            </div>
                          </div>
                        </div>
                        <RangeSlider
                          value={layer.roughness}
                          boundMin={getBounds(index, 'roughness').min}
                          boundMax={getBounds(index, 'roughness').max}
                          min={0}
                          max={150}
                          step={0.5}
                          decimals={1}
                          onValueChange={(v) => updateLayer(index, 'roughness', v)}
                          onBoundsChange={(lo, hi) => setBounds(index, 'roughness', lo, hi)}
                        />
                      </div>

                      <div className="control">
                        <div className="control__label">
                          <span>iSLD<InfoTooltip hint={"Imaginary SLD (absorption).\nTypically 0 for most materials.\n\nDrag ◀ ▶ handles or edit [min] value [max] to set bounds."} /></span>
                          <div className={styles.valueWrapper}>
                            <InfoTooltip hint="Reset iSLD to default">
                              <button
                                className={styles.resetBtn}
                                onClick={() => {
                                  updateLayer(index, 'isld', getLayerDefault(index, 'isld'));
                                  const def = getBoundDefault(index, 'isld');
                                  setBounds(index, 'isld', def.min, def.max);
                                }}
                              >
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                                  <path d="M3 3v5h5" />
                                </svg>
                              </button>
                            </InfoTooltip>
                            <div className={styles.valueWithBounds}>
                              {(getBounds(index, 'isld').min !== null || getBounds(index, 'isld').max !== null) ? (
                                <>
                                  <EditableValue
                                    value={getBounds(index, 'isld').min ?? layer.isld}
                                    onChange={(v) => setBounds(index, 'isld', v, getBounds(index, 'isld').max)}
                                    min={0}
                                    max={layer.isld}
                                    step={0.001}
                                    decimals={3}
                                  />
                                  <EditableValue
                                    value={getBounds(index, 'isld').max ?? layer.isld}
                                    onChange={(v) => setBounds(index, 'isld', getBounds(index, 'isld').min, v)}
                                    min={layer.isld}
                                    max={10}
                                    step={0.001}
                                    decimals={3}
                                  />
                                </>
                              ) : (
                                <EditableValue
                                  value={layer.isld}
                                  onChange={(v) => updateLayer(index, 'isld', v)}
                                  min={0}
                                  max={10}
                                  step={0.001}
                                  decimals={3}
                                />
                              )}
                            </div>
                          </div>
                        </div>
                        <RangeSlider
                          value={layer.isld}
                          boundMin={getBounds(index, 'isld').min}
                          boundMax={getBounds(index, 'isld').max}
                          min={0}
                          max={1}
                          step={0.001}
                          decimals={3}
                          onValueChange={(v) => updateLayer(index, 'isld', v)}
                          onBoundsChange={(lo, hi) => setBounds(index, 'isld', lo, hi)}
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
                <span>Number of Curves<InfoTooltip hint={"How many synthetic NR curves to generate for training.\n\nMore curves = better model but longer training."} />{isProduction && ` (max ${limits.max_curves})`}</span>
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
                <span>Max Film Layers<InfoTooltip hint={"Maximum number of film layers to include in the synthetic data generation.\n\nWhen any layer has a ± range set, this is locked to the actual film layer count."} /></span>
                <EditableValue
                  value={generatorParams.numFilmLayers}
                  onChange={(v) =>
                    onGeneratorParamsChange({
                      ...generatorParams,
                      numFilmLayers: hasAnyBounds
                        ? derivedNumFilmLayers
                        : Math.min(Math.max(1, Math.round(v)), filmLayers.length),
                    })
                  }
                  min={1}
                  max={filmLayers.length}
                  step={1}
                  decimals={0}
                  disabled={hasAnyBounds}
                />
              </div>
              <input
                type="range"
                className="slider"
                min="1"
                max={filmLayers.length}
                step="1"
                value={generatorParams.numFilmLayers}
                onChange={(e) =>
                  onGeneratorParamsChange({
                    ...generatorParams,
                    numFilmLayers: hasAnyBounds
                      ? derivedNumFilmLayers
                      : Math.min(parseInt(e.target.value), filmLayers.length),
                  })
                }
                disabled={hasAnyBounds}
              />
              {hasAnyBounds && (
                <div className={styles.deltaInfo}>
                  Locked: layer bounds active (numFilmLayers = {derivedNumFilmLayers})
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {shouldShowTraining && (
        <div className="section">
          <div className="section__header">
            <h3 className="section__title">
              Training
              <InfoTooltip hint={trainingTooltip} />
            </h3>
          </div>

          {onGpuChange && (
            <div className="control">
              <div className="control__label">
                <span>GPU<InfoTooltip hint={"GPU tier for Modal cloud training.\nSpeed is relative to T4 (FP16 dense TFLOPS).\n\nT4: $0.59/hr, 16GB, 65 TF, 1×\nL4: $0.80/hr, 24GB, 121 TF, 2×\nA10G: $1.10/hr, 24GB, 125 TF, 2×\nL40S: $1.95/hr, 48GB, 362 TF, 5.5×\nA100: $2.10/hr, 40GB, 312 TF, 5×\nA100-80: $2.50/hr, 80GB, 312 TF, 5×\nH100: $3.95/hr, 80GB, 989 TF, 15×\nH200: $4.54/hr, 141GB, 989 TF, 15×\nB200: $6.25/hr, 192GB, 2250 TF, 35×"} /></span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-primary)' }}>
                  {GPU_TIERS.find(g => g.value === gpu)?.description || gpu}
                </span>
              </div>
              <select
                className="control__input"
                value={gpu}
                onChange={(e) => onGpuChange(e.target.value as GpuTier)}
              >
                {GPU_TIERS.map((tier) => (
                  <option key={tier.value} value={tier.value}>
                    {tier.label} — {tier.speed} Speed
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="control">
            <div className="control__label">
              <span>Batch Size<InfoTooltip hint={"Number of samples processed together in one training step.\n\nLarger = faster but uses more memory."} />{isProduction && ` (max ${limits.max_batch_size})`}</span>
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
              <span>Epochs<InfoTooltip hint={"Complete passes through the training data.\n\nMore epochs = better fit but may overfit."} />{isProduction && ` (max ${limits.max_epochs})`}</span>
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
              <span>CNN Layers<InfoTooltip hint={"Depth of the convolutional neural network.\n\nMore layers = more capacity to learn complex patterns."} />{isProduction && ` (max ${limits.max_cnn_layers})`}</span>
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
              <span>Dropout<InfoTooltip hint={"Regularization to prevent overfitting.\n\nHigher = more regularization, may reduce accuracy."} />{isProduction && ` (max ${limits.max_dropout})`}</span>
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
              <span>Latent Dimension<InfoTooltip hint={"Size of the compressed representation in the autoencoder.\n\nLarger = more info retained."} />{isProduction && ` (max ${limits.max_latent_dim})`}</span>
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
              <span>AE Epochs<InfoTooltip hint={"Autoencoder training epochs.\n\nThe AE learns to compress NR curves into latent space."} />{isProduction && ` (max ${limits.max_ae_epochs})`}</span>
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
              <span>MLP Epochs<InfoTooltip hint={"Multi-layer perceptron epochs.\n\nThe MLP maps latent space to SLD profiles."} />{isProduction && ` (max ${limits.max_mlp_epochs})`}</span>
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
      )}

      {/* Data Upload */}
      {dataSource === 'real' && (
        <div className="section">
          <div className="section__header">
            <h3 className="section__title">Data & Models</h3>
          </div>

          <div className={styles.uploadArea}>
            {requiredUploads.length > 0 && (
              <div className={styles.requirements}>
                <div className={styles.requirementsLabel}>
                  Required uploads
                  <InfoTooltip hint="Click any missing item to upload the specific file used by this pipeline and mode." />
                </div>
                <div className={styles.requirementsList}>
                  {requiredUploads.map((role) => {
                    const hasFile = hasSettingFile(role);
                    return (
                      <button
                        key={role}
                        type="button"
                        className={`${styles.requirementButton} ${hasFile ? styles.requirementOk : styles.requirementMissing}`}
                        onClick={() => handleRoleUploadClick(role)}
                        disabled={hasFile || isUploading}
                        title={hasFile ? 'Found in settings.yml' : 'Click to upload'}
                      >
                        {uploadRequirementLabels[role]}
                      </button>
                    );
                  })}
                </div>
                {!backendStatus && (
                  <div className={styles.requirementsHint}>Waiting for backend status...</div>
                )}
                {backendStatus && missingUploads.length > 0 && (
                  <div className={styles.requirementsHint}>{generateBlockReason}</div>
                )}
                {backendStatus && missingUploads.length === 0 && (
                  <div className={styles.requirementsHint}>All required files are set in settings.yml.</div>
                )}
                {(workflow === 'nr_sld' || workflow === 'nr_sld_chi') && nrSldMode === 'train' && (
                  <div className={styles.requirementsToggle}>
                    <label className={styles.toggleLabel}>
                      <input
                        type="checkbox"
                        className={styles.toggleInput}
                        checked={autoGenerateModelStats}
                        onChange={(e) => onAutoGenerateModelStatsChange(e.target.checked)}
                      />
                      <span>Auto-generate model + stats</span>
                      <InfoTooltip hint="If missing, NR→SLD training generates a model (.pth) and normalization stats (.npy) from nr_train/sld_train and saves them to settings.yml paths." />
                    </label>
                  </div>
                )}
              </div>
            )}

            <div className={styles.mapping}>
              <div className={styles.mappingLabel}>
                Current file mapping
                <InfoTooltip hint="Shows the settings.yml path for each role and whether the file exists on disk." />
              </div>
              {pipelineRoles.map((role) => {
                const path = getSettingPath(role);
                const exists = hasSettingFile(role);
                return (
                  <div key={role} className={styles.mappingRow}>
                    <span className={styles.mappingKey}>
                      {stripLabelSuffix(uploadRequirementLabels[role])}
                    </span>
                    <span className={`${styles.mappingValue} ${exists ? styles.mappingOk : styles.mappingMissing}`}>
                      {exists ? (path ? stripKnownExtension(path) : 'Not set') : 'DNE'}
                    </span>
                  </div>
                );
              })}
            </div>

            <input
              ref={uploadInputRef}
              type="file"
              accept={pendingUploadRole ? roleAcceptMap[pendingUploadRole] : '.npy,.pth,.pt'}
              className={styles.fileInputHidden}
              onChange={handleRoleFileChange}
              disabled={isUploading}
            />

            {backendStatus && (
              backendStatus.data_files.length > 0 ||
              backendStatus.curve_files.length > 0 ||
              backendStatus.expt_files.length > 0 ||
              (backendStatus.model_files && backendStatus.model_files.length > 0)
            ) && (
                <div className={styles.availableFiles}>
                  <div className={styles.availableFilesLabel}>Available:</div>
                  {backendStatus.data_files.map((f) => (
                    <span key={f} className={styles.availableFile}>{f}</span>
                  ))}
                  {backendStatus.curve_files.map((f) => (
                    <span key={f} className={styles.availableFile}>{f}</span>
                  ))}
                  {backendStatus.expt_files.map((f) => (
                    <span key={f} className={styles.availableFile}>{f}</span>
                  ))}
                  {backendStatus.model_files?.map((f) => (
                    <span key={f} className={styles.availableFile}>{f}</span>
                  ))}
                </div>
              )}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="section">
        <div className={styles.actionRow}>
          <button
            className="btn btn--outline"
            onClick={() => {
              onReset();
              resetAllLayers();
            }}
            type="button"
            disabled={isUploading}
          >
            RESET
          </button>
          <button
            className={`btn btn--full ${!canGenerate ? styles.generateBlocked : ''}`}
            onClick={handleGenerateClick}
            disabled={isUploading || !canGenerate}
            title={canGenerate ? undefined : generateBlockReason}
          >
            GENERATE
          </button>
        </div>
      </div>


      {showNamePopup && (
        <div className={styles.popupOverlay}>
          <div className={styles.popupContent}>
            <div className={styles.popupTitle}>Generation Name</div>
            <input
              type="text"
              className={styles.popupInput}
              placeholder="e.g. Test Run 1 (max 20 chars)"
              value={generationName}
              maxLength={20}
              onChange={(e) => setGenerationName(e.target.value)}
              autoFocus
              onKeyDown={(e) => e.key === 'Enter' && handleStartGeneration()}
            />
            <div className={styles.popupActions}>
              <button className="btn btn--outline" onClick={handleCancelGeneration}>
                CANCEL
              </button>
              <button className="btn" onClick={handleStartGeneration}>
                START
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
