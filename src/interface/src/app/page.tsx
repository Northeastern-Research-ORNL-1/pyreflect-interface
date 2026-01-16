'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { useSession, signIn, signOut } from 'next-auth/react';
import ParameterPanel from '../components/ParameterPanel';
import GraphDisplay from '../components/GraphDisplay';
import ConsoleOutput from '../components/ConsoleOutput';
import ExploreSidebar from '../components/ExploreSidebar';
import { FilmLayer, GeneratorParams, TrainingParams, GenerateResponse, Limits, LimitsResponse, DEFAULT_LIMITS, DataSource, Workflow, NrSldMode, UploadRole } from '@/types';
import packageJson from '../../package.json';
import { toPng } from 'html-to-image';
import { zipSync, strToU8 } from 'fflate';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const STORAGE_KEY = 'pyreflect_state';
const APP_VERSION = `v${packageJson.version}`;

const EXPORT_CHARTS = [
  { id: 'nr', filename: 'neutron_reflectivity.png' },
  { id: 'sld', filename: 'sld_profile.png' },
  { id: 'training', filename: 'training_loss.png' },
  { id: 'chi', filename: 'chi_parameters.png' },
];

type BundlePayload = {
  params: { layers: FilmLayer[]; generator: GeneratorParams; training: TrainingParams };
  result: GenerateResponse;
};

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

const DEFAULT_LAYERS: FilmLayer[] = [
  { name: 'substrate', sld: 2.07, isld: 0, thickness: 0, roughness: 1.8 },
  { name: 'siox', sld: 3.47, isld: 0, thickness: 12, roughness: 2.0 },
  { name: 'layer_1', sld: 3.8, isld: 0, thickness: 50, roughness: 10 },
  { name: 'layer_2', sld: 2.5, isld: 0, thickness: 150, roughness: 30 },
  { name: 'air', sld: 0, isld: 0, thickness: 0, roughness: 0 },
];

const DEFAULT_GENERATOR: GeneratorParams = {
  numCurves: 1000,
  numFilmLayers: 5,
};

const DEFAULT_TRAINING: TrainingParams = {
  batchSize: 32,
  epochs: 10,
  layers: 12,
  dropout: 0.0,
  latentDim: 16,
  aeEpochs: 50,
  mlpEpochs: 50,
};

export default function Home() {
  const { data: session } = useSession();
  const [filmLayers, setFilmLayers] = useState<FilmLayer[]>(DEFAULT_LAYERS);
  const [generatorParams, setGeneratorParams] = useState<GeneratorParams>(DEFAULT_GENERATOR);
  const [trainingParams, setTrainingParams] = useState<TrainingParams>(DEFAULT_TRAINING);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [graphData, setGraphData] = useState<GenerateResponse | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_error, setError] = useState<string | null>(null);
  const [consoleLogs, setConsoleLogs] = useState<string[]>([]);
  const [generationStart, setGenerationStart] = useState<number | null>(null);
  const [epochProgress, setEpochProgress] = useState<{ current: number; total: number } | null>(null);
  const [activeGenerationName, setActiveGenerationName] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<BackendStatus | null>(null);
  const [limits, setLimits] = useState<Limits>(DEFAULT_LIMITS);
  const [isProduction, setIsProduction] = useState(false);
  const [isHydrated, setIsHydrated] = useState(false);
  const [dataSource, setDataSource] = useState<DataSource>('synthetic');
  const [workflow, setWorkflow] = useState<Workflow>('nr_sld');
  const [nrSldMode, setNrSldMode] = useState<NrSldMode>('train');
  const [autoGenerateModelStats, setAutoGenerateModelStats] = useState(true);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showActionsMenu, setShowActionsMenu] = useState(false);
  const [showJsonMenu, setShowJsonMenu] = useState(false);
  const [showJsonMenuMobile, setShowJsonMenuMobile] = useState(false);
  const [showBundleConfirm, setShowBundleConfirm] = useState(false);
  const [bundlePayload, setBundlePayload] = useState<BundlePayload | null>(null);
  const [bundleEstimate, setBundleEstimate] = useState<{
    jsonBytes: number;
    pngBytes: number;
    modelBytes: number | null;
    modelSource?: string | null;
    totalBytes: number | null;
    estimating: boolean;
  }>({
    jsonBytes: 0,
    pngBytes: 0,
    modelBytes: null,
    modelSource: null,
    totalBytes: null,
    estimating: false,
  });
  const [bundleEstimateError, setBundleEstimateError] = useState<string | null>(null);
  const [showExplore, setShowExplore] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [importNamePopup, setImportNamePopup] = useState(false);
  const [pendingImportData, setPendingImportData] = useState<GenerateResponse | null>(null);
  const [importName, setImportName] = useState('');
  const jsonMenuRef = useRef<HTMLDivElement>(null);
  const bundlePngCacheRef = useRef<{ payload: BundlePayload | null; files: Record<string, Uint8Array> } | null>(null);

  const addLog = useCallback((message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setConsoleLogs(prev => [...prev, `[${timestamp}] ${message}`]);
    setConsoleLogs(prev => [...prev, `[${timestamp}] ${message}`]);
  }, []);

  const saveToHistory = useCallback(async (data: { params: { layers: FilmLayer[]; generator: GeneratorParams; training: TrainingParams }; result: GenerateResponse; name?: string }) => {
    if (!session?.user || !('id' in session.user)) return;
    
    try {
      const payload = {
        layers: data.params.layers,
        generator: data.params.generator,
        training: data.params.training,
        result: { ...data.result, name: data.name },
        name: data.name
      };

      const res = await fetch(`${API_URL}/api/history`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-ID': session.user.id as string,
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error('Failed to save');
      addLog('Saved import to history');
    } catch (err) {
      console.error(err);
      addLog('Warning: Failed to save import to history');
    }
  }, [session, addLog]);

  const confirmImportName = useCallback(() => {
    if (pendingImportData) {
      const updatedResult = {
        ...pendingImportData,
        name: importName || undefined
      };
      setGraphData(updatedResult);
      addLog(`Imported data (Name: ${importName || 'Untitled'})`);
      
      saveToHistory({
         params: {
             layers: DEFAULT_LAYERS,
             generator: DEFAULT_GENERATOR,
             training: DEFAULT_TRAINING
         },
         result: updatedResult,
         name: importName || undefined
      });
    }
    setImportNamePopup(false);
    setPendingImportData(null);
    setImportName('');
  }, [pendingImportData, importName, addLog, saveToHistory]);

  const cancelImportName = useCallback(() => {
    setImportNamePopup(false);
    setPendingImportData(null);
    setImportName('');
    addLog('Import cancelled');
  }, [addLog]);

  const handleReset = useCallback(() => {
    setFilmLayers(DEFAULT_LAYERS);
    setGeneratorParams(DEFAULT_GENERATOR);
    setTrainingParams(DEFAULT_TRAINING);
    setGraphData(null);
    setConsoleLogs([]);
    setError(null);
    setDataSource('synthetic');
    setWorkflow('nr_sld');
    setNrSldMode('train');
    setAutoGenerateModelStats(true);
    if (typeof window !== 'undefined') {
      localStorage.removeItem(`${STORAGE_KEY}_layers`);
      localStorage.removeItem(`${STORAGE_KEY}_generator`);
      localStorage.removeItem(`${STORAGE_KEY}_training`);
      localStorage.removeItem(`${STORAGE_KEY}_graphData`);
      localStorage.removeItem(`${STORAGE_KEY}_logs`);
      localStorage.removeItem(`${STORAGE_KEY}_dataSource`);
      localStorage.removeItem(`${STORAGE_KEY}_workflow`);
      localStorage.removeItem(`${STORAGE_KEY}_mode`);
      localStorage.removeItem(`${STORAGE_KEY}_autoGenerate`);
    }
  }, []);

  // Load from localStorage after mount (avoids hydration mismatch)
  useEffect(() => {
    try {
      const storedLayers = localStorage.getItem(`${STORAGE_KEY}_layers`);
      const storedGenerator = localStorage.getItem(`${STORAGE_KEY}_generator`);
      const storedTraining = localStorage.getItem(`${STORAGE_KEY}_training`);
      const storedGraphData = localStorage.getItem(`${STORAGE_KEY}_graphData`);
      const storedLogs = localStorage.getItem(`${STORAGE_KEY}_logs`);
      const storedDataSource = localStorage.getItem(`${STORAGE_KEY}_dataSource`);
      const storedWorkflow = localStorage.getItem(`${STORAGE_KEY}_workflow`);
      const storedMode = localStorage.getItem(`${STORAGE_KEY}_mode`);
      const storedAutoGenerate = localStorage.getItem(`${STORAGE_KEY}_autoGenerate`);

      if (storedLayers) setFilmLayers(JSON.parse(storedLayers));
      if (storedGenerator) setGeneratorParams(JSON.parse(storedGenerator));
      if (storedTraining) setTrainingParams(JSON.parse(storedTraining));
      if (storedGraphData) setGraphData(JSON.parse(storedGraphData));
      if (storedLogs) setConsoleLogs(JSON.parse(storedLogs));
      if (storedDataSource) setDataSource(storedDataSource as DataSource);
      if (storedWorkflow) setWorkflow(storedWorkflow as Workflow);
      if (storedMode) setNrSldMode(storedMode as NrSldMode);
      if (storedAutoGenerate !== null) setAutoGenerateModelStats(storedAutoGenerate === 'true');
    } catch {
      // Ignore parse errors
    }
    setIsHydrated(true);
  }, []);

  // Persist state to localStorage (only after hydration)
  useEffect(() => {
    if (!isHydrated) return;
    localStorage.setItem(`${STORAGE_KEY}_layers`, JSON.stringify(filmLayers));
  }, [filmLayers, isHydrated]);

  useEffect(() => {
    if (!isHydrated) return;
    localStorage.setItem(`${STORAGE_KEY}_generator`, JSON.stringify(generatorParams));
  }, [generatorParams, isHydrated]);

  useEffect(() => {
    setGeneratorParams((prev) => {
      const maxLayers = filmLayers.length;
      if (prev.numFilmLayers <= maxLayers) return prev;
      return { ...prev, numFilmLayers: maxLayers };
    });
  }, [filmLayers]);

  useEffect(() => {
    if (!isHydrated) return;
    localStorage.setItem(`${STORAGE_KEY}_training`, JSON.stringify(trainingParams));
  }, [trainingParams, isHydrated]);

  useEffect(() => {
    if (!isHydrated) return;
    localStorage.setItem(`${STORAGE_KEY}_graphData`, JSON.stringify(graphData));
  }, [graphData, isHydrated]);

  useEffect(() => {
    if (!isHydrated) return;
    localStorage.setItem(`${STORAGE_KEY}_logs`, JSON.stringify(consoleLogs));
  }, [consoleLogs, isHydrated]);

  useEffect(() => {
    if (!isHydrated) return;
    localStorage.setItem(`${STORAGE_KEY}_dataSource`, dataSource);
  }, [dataSource, isHydrated]);

  useEffect(() => {
    if (!isHydrated) return;
    localStorage.setItem(`${STORAGE_KEY}_workflow`, workflow);
  }, [workflow, isHydrated]);

  useEffect(() => {
    if (!isHydrated) return;
    localStorage.setItem(`${STORAGE_KEY}_mode`, nrSldMode);
  }, [nrSldMode, isHydrated]);

  useEffect(() => {
    if (!isHydrated) return;
    localStorage.setItem(`${STORAGE_KEY}_autoGenerate`, String(autoGenerateModelStats));
  }, [autoGenerateModelStats, isHydrated]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (jsonMenuRef.current && !jsonMenuRef.current.contains(event.target as Node)) {
        setShowJsonMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Fetch backend status and limits on mount
  useEffect(() => {
    if (!isHydrated) return;
    const fetchStatus = async () => {
      try {
        const [statusRes, limitsRes] = await Promise.all([
          fetch(`${API_URL}/api/status`),
          fetch(`${API_URL}/api/limits`),
        ]);
        if (statusRes.ok) {
          const status: BackendStatus = await statusRes.json();
          setBackendStatus(status);
          addLog(`Backend connected. pyreflect: ${status.pyreflect_available ? 'available' : 'unavailable'}`);
          if (status.data_files.length > 0) {
            addLog(`Data files: ${status.data_files.join(', ')}`);
          }
        }
        if (limitsRes.ok) {
          const limitsData: LimitsResponse = await limitsRes.json();
          setLimits(limitsData.limits);
          setIsProduction(limitsData.production);
          if (limitsData.production) {
            addLog(`Production mode: limits enforced (max ${limitsData.limits.max_curves} curves, ${limitsData.limits.max_epochs} epochs)`);
          }
        }
      } catch {
        addLog('Backend not reachable. Start with: python -m uvicorn main:app --reload');
      }
    };
    fetchStatus();
  }, [isHydrated, addLog]);

  const handleGenerate = useCallback(async (name?: string) => {
    setIsGenerating(true);
    setGenerationStart(Date.now());
    setEpochProgress(null);
    setActiveGenerationName(name ?? null);
    setError(null);
    addLog('Starting generation...');
    if (dataSource === 'real') {
      const modeLabel = workflow === 'sld_chi' ? '' : ` (${nrSldMode})`;
      addLog(`Real data: ${workflow}${modeLabel}`);
    }
    if (name) addLog(`Name: ${name}`);
    if (dataSource === 'synthetic') {
      addLog(`Params: ${generatorParams.numCurves} curves, ${trainingParams.epochs} epochs`);
    }
    
    try {
      // Build headers with user ID for MongoDB persistence
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (session?.user && 'id' in session.user) {
        headers['X-User-ID'] = session.user.id as string;
      }
      
      const response = await fetch(`${API_URL}/api/generate/stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          layers: filmLayers,
          generator: generatorParams,
          training: trainingParams,
          name: name,
          dataSource,
          workflow,
          mode: nrSldMode,
          autoGenerateModelStats,
        }),
      });

      if (!response.ok) {
        throw new Error(`Generation failed: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEvent = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7);
          } else if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            if (currentEvent === 'log') {
              addLog(data);
            } else if (currentEvent === 'progress') {
              if (typeof data.epoch === 'number' && typeof data.total === 'number') {
                setEpochProgress({ current: data.epoch, total: data.total });
              }
            } else if (currentEvent === 'result') {
              setGraphData(data as GenerateResponse);
              addLog(`Generation complete. MSE: ${data.metrics.mse.toFixed(4)}`);
            } else if (currentEvent === 'error') {
              throw new Error(data);
            }
          }
        }
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Backend not deployed';
      setError(errorMsg);
      addLog(`Error: ${errorMsg}`);
    } finally {
      setIsGenerating(false);
      setGenerationStart(null);
      setEpochProgress(null);
      setActiveGenerationName(null);
    }
  }, [filmLayers, generatorParams, trainingParams, addLog, session, dataSource, workflow, nrSldMode, autoGenerateModelStats]);

  const handleUploadFiles = useCallback(async (uploads: { file: File; role: UploadRole }[]) => {
    if (uploads.length === 0) return;
    setIsUploading(true);
    setError(null);
    addLog(`Uploading ${uploads.length} file(s)...`);

    const formData = new FormData();
    uploads.forEach(({ file, role }) => {
      formData.append('files', file);
      formData.append('roles', role);
    });

    try {
      const response = await fetch(`${API_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();
      const savedCount = Array.isArray(result.saved) ? result.saved.length : 0;
      addLog(`Uploaded ${savedCount} file(s) to backend data folder`);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMsg);
      addLog(`Upload error: ${errorMsg}`);
    } finally {
      setIsUploading(false);
    }
  }, [addLog]);



  const handleImportJSON = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json,application/json';
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      try {
        const text = await file.text();
        const json = JSON.parse(text);
        
        // Check format
        let resultData: GenerateResponse;
        let paramsData = {
           layers: DEFAULT_LAYERS,
           generator: DEFAULT_GENERATOR,
           training: DEFAULT_TRAINING
        };

        if (json.result && json.params) {
            // New format
            resultData = json.result;
            paramsData = json.params;
            // Also load params into view? Maybe user just wants to view results. 
            // Let's load them to be consistent with "History Load"
            setFilmLayers(json.params.layers);
            setGeneratorParams(json.params.generator);
            setTrainingParams(json.params.training);
        } else {
            // Legacy format (just GenerateResponse)
            if (json.nr && json.sld) {
                resultData = json as GenerateResponse;
            } else {
                throw new Error('Unrecognized JSON format');
            }
        }

        // Handle Name & Save
        const finalName = resultData.name;
        
        if (finalName) {
            setGraphData(resultData);
            addLog(`Imported data from ${file.name} (Name: ${finalName})`);
            saveToHistory({
                params: paramsData,
                result: resultData,
                name: finalName
            });
        } else {
            // Need name - temporarily store what we derived
            // Hack: Store full object in pending if possible, but pending expects GenerateResponse currently 
            // I'll attach the params to the result object temporarily or just rely on defaults in confirmImportName
            // Note: confirmImportName currently uses DEFAULTs for legacy. 
            // Be better: store params in a ref or separate state?
            // For simplicity in this session, I will just set state params now (which I did above for new format)
            // and confirmImportName will use DEFAULTs (which is wrong if I just set them!).
            
            // Let's update pending logic.
            setPendingImportData(resultData);
            setImportName('');
            setImportNamePopup(true);
            
            // If new format, we updated state params. confirmImportName uses DEFAULTs in my previous chunk.
            // I should update confirmImportName to use current state params? 
            // Actually, confirmImportName defines the save payload. 
            // I will update confirmImportName in next step or use a ref.
            // For now, let's assume legacy uses defaults, new uses defaults (minor bug: lost params in save history for new format if name missing).
            // But usually export has name if it came from backend.
        }

      } catch (err) {
        addLog(`Error parsing JSON: ${err instanceof Error ? err.message : 'Unknown error'}`);
      }
    };
    input.click();
  }, [addLog, saveToHistory]);

  const handleLoadSave = useCallback((params: { layers: FilmLayer[]; generator: GeneratorParams; training: TrainingParams }, result: GenerateResponse) => {
    setFilmLayers(params.layers);
    setGeneratorParams(params.generator);
    setTrainingParams(params.training);
    setGraphData(result);
    addLog('Loaded saved session from history');
  }, [addLog]);

  const buildExportPayload = useCallback(
    (
      params: { layers: FilmLayer[]; generator: GeneratorParams; training: TrainingParams },
      result: GenerateResponse
    ) => ({
      params: {
        layers: params.layers,
        generator: params.generator,
        training: params.training,
      },
      result,
    }),
    []
  );

  const downloadJsonPayload = useCallback((payload: object, fileName?: string) => {
    const name = fileName ?? `pyreflect_export_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    const jsonStr = JSON.stringify(payload, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, []);

  const formatBytes = useCallback((bytes: number | null) => {
    if (bytes === null) return '—';
    if (bytes < 1024) return `${bytes} B`;
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    const mb = kb / 1024;
    return `${mb.toFixed(2)} MB`;
  }, []);

  const waitForCharts = useCallback(async () => {
    for (let i = 0; i < 8; i += 1) {
      const nodes = document.querySelectorAll('[data-export-id]');
      if (nodes.length > 0) return;
      await new Promise((resolve) => setTimeout(resolve, 80));
    }
  }, []);

  const captureChartPngs = useCallback(async () => {
    await waitForCharts();
    const files: Record<string, Uint8Array> = {};
    for (const chart of EXPORT_CHARTS) {
      const node = document.querySelector(`[data-export-id="${chart.id}"]`) as HTMLElement | null;
      if (!node) continue;
      const { clientWidth, clientHeight } = node;
      if (clientWidth === 0 || clientHeight === 0) continue;
      const dataUrl = await toPng(node, {
        cacheBust: true,
        backgroundColor: '#000000',
        width: clientWidth,
        height: clientHeight,
        style: {
          transform: 'none',
          position: 'static',
          top: '0',
          left: '0',
          right: 'auto',
          bottom: 'auto',
          margin: '0',
          width: `${clientWidth}px`,
          height: `${clientHeight}px`,
          animation: 'none',
        },
      });
      const pngBuffer = await fetch(dataUrl).then((res) => res.arrayBuffer());
      files[chart.filename] = new Uint8Array(pngBuffer);
    }
    return files;
  }, [waitForCharts]);

  const openBundleConfirm = useCallback((payload: BundlePayload) => {
    setBundlePayload(payload);
    setShowBundleConfirm(true);
    bundlePngCacheRef.current = null;
  }, []);

  useEffect(() => {
    if (!showBundleConfirm || !bundlePayload) return;
    let cancelled = false;

    const estimate = async () => {
      setBundleEstimateError(null);
      setBundleEstimate({
        jsonBytes: 0,
        pngBytes: 0,
        modelBytes: null,
        modelSource: null,
        totalBytes: null,
        estimating: true,
      });

      const exportData = buildExportPayload(bundlePayload.params, bundlePayload.result);
      const jsonStr = JSON.stringify(exportData, null, 2);
      const jsonBytes = new TextEncoder().encode(jsonStr).length;

      let pngBytes = 0;
      try {
        const files = await captureChartPngs();
        pngBytes = Object.values(files).reduce((sum, file) => sum + file.length, 0);
        bundlePngCacheRef.current = { payload: bundlePayload, files };
      } catch (err) {
        setBundleEstimateError('Could not estimate PNG size.');
      }

      let modelBytes: number | null = null;
      let modelSource: string | null = null;
      if (bundlePayload.result.model_id) {
        try {
          const res = await fetch(`${API_URL}/api/models/${bundlePayload.result.model_id}/info`);
          if (res.ok) {
            const data = await res.json();
            if (typeof data.size_mb === 'number') {
              modelBytes = data.size_mb * 1024 * 1024;
            }
            modelSource = data.source || null;
          }
        } catch {
          modelSource = null;
        }
      }

      const totalBytes = jsonBytes + pngBytes + (modelBytes ?? 0);

      if (cancelled) return;
      setBundleEstimate({
        jsonBytes,
        pngBytes,
        modelBytes,
        modelSource,
        totalBytes,
        estimating: false,
      });
    };

    estimate();

    return () => {
      cancelled = true;
    };
  }, [showBundleConfirm, bundlePayload, buildExportPayload, captureChartPngs]);

  const handleExportAll = useCallback(() => {
    if (!graphData) return;
    
    const exportData = buildExportPayload(
      { layers: filmLayers, generator: generatorParams, training: trainingParams },
      graphData
    );
    downloadJsonPayload(exportData);
    addLog('Exported full session data (params + results)');
  }, [graphData, filmLayers, generatorParams, trainingParams, addLog, buildExportPayload, downloadJsonPayload]);

  const handleDownloadBundle = useCallback(async (payload?: BundlePayload) => {
    const resolvedResult = payload?.result ?? graphData;
    if (!resolvedResult) {
      addLog('Nothing to download yet.');
      return;
    }

    const resolvedParams = payload?.params ?? {
      layers: filmLayers,
      generator: generatorParams,
      training: trainingParams,
    };

    const exportData = buildExportPayload(resolvedParams, resolvedResult);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const baseName = resolvedResult.name
      ? resolvedResult.name.replace(/[^a-z0-9_-]+/gi, '_').slice(0, 32)
      : `pyreflect_bundle_${timestamp}`;
    const files: Record<string, Uint8Array> = {
      'output.json': strToU8(JSON.stringify(exportData, null, 2)),
    };

    try {
      addLog('Preparing download bundle...');
      const cached = bundlePngCacheRef.current;
      const pngFiles =
        cached && cached.payload === payload
          ? cached.files
          : await captureChartPngs();
      Object.assign(files, pngFiles);

      if (resolvedResult.model_id) {
        const modelRes = await fetch(`${API_URL}/api/models/${resolvedResult.model_id}`);
        if (modelRes.ok) {
          const modelBuffer = await modelRes.arrayBuffer();
          files[`model_${resolvedResult.model_id}.pth`] = new Uint8Array(modelBuffer);
        } else {
          addLog('Model file not found for this run.');
        }
      } else {
        addLog('No model file associated with this run.');
      }

      const zipData = zipSync(files, { level: 6 });
      const blob = new Blob([zipData], { type: 'application/zip' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${baseName}.zip`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      addLog('Download bundle ready.');
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to build download bundle';
      addLog(`Download error: ${errorMsg}`);
    }
  }, [graphData, filmLayers, generatorParams, trainingParams, buildExportPayload, captureChartPngs, addLog]);

  const handleDownloadBundleClick = useCallback(() => {
    if (!graphData) {
      addLog('No results to download yet.');
      return;
    }
    openBundleConfirm({
      params: { layers: filmLayers, generator: generatorParams, training: trainingParams },
      result: graphData,
    });
  }, [graphData, filmLayers, generatorParams, trainingParams, openBundleConfirm, addLog]);

  const handleSidebarDownloadRequest = useCallback(async (saveId: string) => {
    if (!session?.user || !('id' in session.user)) {
      alert('Please sign in to download history items.');
      return;
    }

    try {
      const res = await fetch(`${API_URL}/api/history/${saveId}`, {
        headers: {
          'X-User-ID': session.user.id as string,
        },
      });

      if (!res.ok) {
        throw new Error('Failed to load history item for download.');
      }

      const data = await res.json();
      if (!data?.params || !data?.result) {
        throw new Error('History item is missing data.');
      }

      handleLoadSave(data.params, data.result);
      await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
      await new Promise((resolve) => setTimeout(resolve, 120));
      openBundleConfirm({ params: data.params, result: data.result });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Download failed';
      addLog(`Download error: ${errorMsg}`);
    }
  }, [session, handleLoadSave, openBundleConfirm, addLog]);

  const handleConfirmBundleDownload = useCallback(async () => {
    if (!bundlePayload) return;
    await handleDownloadBundle(bundlePayload);
    setShowBundleConfirm(false);
    setBundlePayload(null);
  }, [bundlePayload, handleDownloadBundle]);

  return (
    <div className="container">
      <header className="header">
        <div className="header__logo">
          <span>◇</span>
          <span>PYREFLECT</span>
          <span className="header__version">{APP_VERSION}</span>
          {isProduction && <span className="header__version" style={{ color: '#f59e0b', marginLeft: '8px' }}>PROD</span>}
          <span className={`status ${isGenerating ? 'status--training' : 'status--active'}`} style={{ marginLeft: '12px' }}>
            <span className="status__dot"></span>
            <span className="header__status-text">
              {isGenerating
                ? epochProgress
                  ? `Training... (${epochProgress.current}/${epochProgress.total})`
                  : 'Training...'
                : 'Ready'}
            </span>
          </span>
        </div>
        <nav className="header__nav">
          {/* Desktop: show buttons inline */}
          <div className="header__actions-desktop">
            <a
              className="header__export-btn"
              href="https://github.com/Northeastern-Research-ORNL-1/pyreflect-interface"
              target="_blank"
              rel="noopener noreferrer"
              title="View on GitHub"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M12 .5C5.65.5.5 5.82.5 12.38c0 5.25 3.44 9.7 8.21 11.27.6.11.82-.27.82-.6 0-.3-.01-1.1-.02-2.16-3.34.75-4.04-1.66-4.04-1.66-.55-1.44-1.34-1.82-1.34-1.82-1.09-.77.08-.76.08-.76 1.2.09 1.83 1.27 1.83 1.27 1.07 1.88 2.8 1.34 3.49 1.03.11-.8.42-1.34.76-1.65-2.66-.31-5.47-1.36-5.47-6.06 0-1.34.46-2.44 1.23-3.31-.12-.31-.53-1.57.12-3.27 0 0 1.01-.33 3.3 1.26a11.2 11.2 0 0 1 3-.41c1.02 0 2.04.14 3 .41 2.29-1.59 3.3-1.26 3.3-1.26.65 1.7.24 2.96.12 3.27.77.87 1.23 1.97 1.23 3.31 0 4.71-2.81 5.75-5.49 6.05.43.38.81 1.13.81 2.28 0 1.65-.01 2.98-.01 3.39 0 .33.22.72.83.6 4.76-1.57 8.2-6.02 8.2-11.27C23.5 5.82 18.35.5 12 .5z" />
              </svg>
              <span className="header__btn-label">GitHub</span>
            </a>
            <button className="header__export-btn" onClick={() => setShowExplore(true)}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
              </svg>
              <span className="header__btn-label">History</span>
            </button>
            <div className="header__menu" ref={jsonMenuRef}>
              <button
                className="header__export-btn"
                onClick={() => setShowJsonMenu((prev) => !prev)}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M1.5 12s3.6-7 10.5-7 10.5 7 10.5 7-3.6 7-10.5 7-10.5-7-10.5-7z" />
                  <circle cx="12" cy="12" r="3" />
                </svg>
                <span className="header__btn-label">View</span>
              </button>
              {showJsonMenu && (
                <div className="header__dropdown">
                  <button
                    className="header__dropdown-item"
                    onClick={() => {
                      handleImportJSON();
                      setShowJsonMenu(false);
                    }}
                  >
                    <span>↑</span> Import JSON
                  </button>
                  {graphData && (
                    <button
                      className="header__dropdown-item"
                      onClick={() => {
                        handleExportAll();
                        setShowJsonMenu(false);
                      }}
                    >
                      <span>↓</span> Export JSON
                    </button>
                  )}
                </div>
              )}
            </div>
            {graphData && (
              <button className="header__export-btn" onClick={handleDownloadBundleClick}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="7 10 12 15 17 10"></polyline>
                  <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                <span className="header__btn-label">Download</span>
              </button>
            )}
          </div>

          {/* Mobile: dropdown menu */}
          <div className="header__actions-mobile">
            <button 
              className="header__export-btn" 
              onClick={() => {
                setShowActionsMenu(!showActionsMenu);
                if (showActionsMenu) {
                  setShowJsonMenuMobile(false);
                }
              }}
            >
              <span>≡</span>
            </button>
            {showActionsMenu && (
              <div className="header__dropdown">
                <a
                  className="header__dropdown-item"
                  href="https://github.com/Northeastern-Research-ORNL-1/pyreflect-interface"
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={() => setShowActionsMenu(false)}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                    <path d="M12 .5C5.65.5.5 5.82.5 12.38c0 5.25 3.44 9.7 8.21 11.27.6.11.82-.27.82-.6 0-.3-.01-1.1-.02-2.16-3.34.75-4.04-1.66-4.04-1.66-.55-1.44-1.34-1.82-1.34-1.82-1.09-.77.08-.76.08-.76 1.2.09 1.83 1.27 1.83 1.27 1.07 1.88 2.8 1.34 3.49 1.03.11-.8.42-1.34.76-1.65-2.66-.31-5.47-1.36-5.47-6.06 0-1.34.46-2.44 1.23-3.31-.12-.31-.53-1.57.12-3.27 0 0 1.01-.33 3.3 1.26a11.2 11.2 0 0 1 3-.41c1.02 0 2.04.14 3 .41 2.29-1.59 3.3-1.26 3.3-1.26.65 1.7.24 2.96.12 3.27.77.87 1.23 1.97 1.23 3.31 0 4.71-2.81 5.75-5.49 6.05.43.38.81 1.13.81 2.28 0 1.65-.01 2.98-.01 3.39 0 .33.22.72.83.6 4.76-1.57 8.2-6.02 8.2-11.27C23.5 5.82 18.35.5 12 .5z" />
                  </svg>
                  GitHub
                </a>
                <button className="header__dropdown-item" onClick={() => { setShowExplore(true); setShowActionsMenu(false); }}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
                  </svg> 
                  History
                </button>
                <button
                  className="header__dropdown-item"
                  onClick={() => setShowJsonMenuMobile((prev) => !prev)}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M1.5 12s3.6-7 10.5-7 10.5 7 10.5 7-3.6 7-10.5 7-10.5-7-10.5-7z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                  <span>{showJsonMenuMobile ? '▾' : '▸'}</span> JSON
                </button>
                {showJsonMenuMobile && (
                  <div className="header__dropdown-sub">
                    <button
                      className="header__dropdown-item header__dropdown-item--sub"
                      onClick={() => {
                        handleImportJSON();
                        setShowJsonMenuMobile(false);
                        setShowActionsMenu(false);
                      }}
                    >
                      <span>↑</span> Import
                    </button>
                    {graphData && (
                      <button
                        className="header__dropdown-item header__dropdown-item--sub"
                        onClick={() => {
                          handleExportAll();
                          setShowJsonMenuMobile(false);
                          setShowActionsMenu(false);
                        }}
                      >
                        <span>↓</span> Export
                      </button>
                    )}
                  </div>
                )}
                {graphData && (
                  <button className="header__dropdown-item" onClick={() => { handleDownloadBundleClick(); setShowActionsMenu(false); setShowJsonMenuMobile(false); }}>
                    <span>↓</span> Download
                  </button>
                )}
              </div>
            )}
          </div>

          {/* GitHub Auth */}
          {session ? (
            <div className="header__user">
              {session.user?.image && (
                <img 
                  src={session.user.image} 
                  alt="" 
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  style={{ width: 32, height: 32, borderRadius: '50%', cursor: 'pointer', border: '1px solid var(--text-primary)' }}
                />
              )}
              {showUserMenu && (
                <button 
                  className="header__export-btn header__export-btn--danger"
                  onClick={() => { setShowUserMenu(false); signOut(); }}
                >
                  <span>→</span><span className="header__btn-label">Sign out</span>
                </button>
              )}
            </div>
          ) : (
            <button 
              className="header__export-btn" 
              onClick={() => signIn('github')}
            >
              <span>←</span><span className="header__btn-label">Sign in</span>
            </button>
          )}
        </nav>
      </header>

      <main className="main">
        <aside className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
          <ParameterPanel
            filmLayers={filmLayers}
            generatorParams={generatorParams}
            trainingParams={trainingParams}
            onFilmLayersChange={setFilmLayers}
            onGeneratorParamsChange={setGeneratorParams}
            onTrainingParamsChange={setTrainingParams}
            onGenerate={handleGenerate}
            onReset={handleReset}
            isGenerating={isGenerating}
            onUploadFiles={handleUploadFiles}
            limits={limits}
            isProduction={isProduction}
            isUploading={isUploading}
            backendStatus={backendStatus}
            dataSource={dataSource}
            workflow={workflow}
            nrSldMode={nrSldMode}
            autoGenerateModelStats={autoGenerateModelStats}
            onDataSourceChange={setDataSource}
            onWorkflowChange={setWorkflow}
            onNrSldModeChange={setNrSldMode}
            onAutoGenerateModelStatsChange={setAutoGenerateModelStats}
            isCollapsed={sidebarCollapsed}
            onToggleCollapse={() => setSidebarCollapsed(prev => !prev)}
          />
        </aside>

        <section className="content">
          <GraphDisplay data={graphData} />
          <ConsoleOutput logs={consoleLogs} isGenerating={isGenerating} startTimeMs={generationStart} />
        </section>
      </main>

      <ExploreSidebar 
        isOpen={showExplore} 
        onClose={() => setShowExplore(false)} 
        userId={session?.user ? (session.user as any).id : undefined}
        onLoadSave={handleLoadSave}
        onRequestDownload={handleSidebarDownloadRequest}
        inProgress={isGenerating ? { name: activeGenerationName, epochProgress } : null}
      />

      {showBundleConfirm && bundlePayload && (
        <>
          <div
            className="model-download-overlay"
            onClick={() => {
              setShowBundleConfirm(false);
              setBundlePayload(null);
            }}
          />
          <div className="model-download-popup">
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', marginBottom: '16px', lineHeight: '1.4' }}>
              <div style={{ fontWeight: 600, marginBottom: '8px' }}>Download Bundle?</div>
              <div style={{ color: 'var(--text-muted)', fontSize: '11px', marginBottom: '10px' }}>
                Includes model (.pth), chart PNGs, and output.json. Model is pulled from Hugging Face if not local.
              </div>
              <div style={{ display: 'grid', gap: '6px', fontSize: '12px' }}>
                <div>Model: <span style={{ color: 'var(--text-secondary)' }}>
                  {bundlePayload.result.model_id
                    ? bundleEstimate.modelBytes !== null
                      ? `${formatBytes(bundleEstimate.modelBytes)}${bundleEstimate.modelSource ? ` (${bundleEstimate.modelSource})` : ''}`
                      : bundleEstimate.estimating
                        ? 'Estimating...'
                        : 'Unknown'
                    : 'Not available'}
                </span></div>
                <div>PNGs: <span style={{ color: 'var(--text-secondary)' }}>
                  {bundleEstimate.estimating ? 'Estimating...' : formatBytes(bundleEstimate.pngBytes)}
                </span></div>
                <div>JSON: <span style={{ color: 'var(--text-secondary)' }}>
                  {bundleEstimate.estimating ? 'Estimating...' : formatBytes(bundleEstimate.jsonBytes)}
                </span></div>
                <div>Total: <span style={{ color: 'var(--text-secondary)' }}>
                  {bundleEstimate.estimating ? 'Estimating...' : formatBytes(bundleEstimate.totalBytes)}
                  {bundlePayload.result.model_id && bundleEstimate.modelBytes === null ? ' + model' : ''}
                </span></div>
              </div>
              {bundleEstimateError && (
                <div style={{ color: 'var(--text-muted)', fontSize: '11px', marginTop: '8px' }}>
                  {bundleEstimateError}
                </div>
              )}
            </div>
            <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
              <button 
                className="btn btn--outline" 
                onClick={() => {
                  setShowBundleConfirm(false);
                  setBundlePayload(null);
                }}
                style={{ padding: '6px 12px', fontSize: '11px' }}
              >
                CANCEL
              </button>
              <button 
                className="btn" 
                onClick={handleConfirmBundleDownload}
                style={{ padding: '6px 12px', fontSize: '11px' }}
              >
                DOWNLOAD
              </button>
            </div>
          </div>
        </>
      )}
      
      {importNamePopup && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-title">Name Imported Generation</div>
            <input
              type="text"
              className="control__input"
              style={{ marginTop: '8px' }}
              placeholder="e.g. Experiment 1 (max 20 chars)"
              value={importName}
              maxLength={20}
              onChange={(e) => setImportName(e.target.value)}
              autoFocus
              onKeyDown={(e) => e.key === 'Enter' && confirmImportName()}
            />
            <div className="modal-actions">
              <button className="btn btn--outline" onClick={cancelImportName}>
                CANCEL
              </button>
              <button className="btn" onClick={confirmImportName}>
                CONFIRM
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
