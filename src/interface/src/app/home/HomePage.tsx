'use client';

import { useCallback, useEffect, useState } from 'react';
import { useSession } from 'next-auth/react';

import AppHeader from '@/components/AppHeader';
import DownloadBundleModal from '@/components/DownloadBundleModal';
import ImportNameModal from '@/components/ImportNameModal';
import ParameterPanel from '@/components/ParameterPanel';
import GraphDisplay from '@/components/GraphDisplay';
import ConsoleOutput from '@/components/ConsoleOutput';
import ExploreSidebar from '@/components/ExploreSidebar';
import {
  DEFAULT_LIMITS,
  DataSource,
  ExportPngs,
  FilmLayer,
  GenerateResponse,
  GeneratorParams,
  Limits,
  LimitsResponse,
  NrSldMode,
  TrainingParams,
  UploadRole,
  Workflow,
} from '@/types';

import packageJson from '../../../package.json';
import { useDownloadBundle, type BundleParams } from './useDownloadBundle';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const STORAGE_KEY = 'pyreflect_state';
const APP_VERSION = `v${packageJson.version}`;

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

type PendingImport = {
  params: BundleParams;
  result: GenerateResponse;
  exportPngs: ExportPngs | null;
};

export default function HomePage() {
  const { data: session } = useSession();
  const sessionUserId = (() => {
    const user = session?.user as unknown;
    if (!user || typeof user !== 'object') return undefined;
    const maybeId = (user as Record<string, unknown>).id;
    return typeof maybeId === 'string' ? maybeId : undefined;
  })();

  const [filmLayers, setFilmLayers] = useState<FilmLayer[]>(DEFAULT_LAYERS);
  const [generatorParams, setGeneratorParams] = useState<GeneratorParams>(DEFAULT_GENERATOR);
  const [trainingParams, setTrainingParams] = useState<TrainingParams>(DEFAULT_TRAINING);
  const [dataSource, setDataSource] = useState<DataSource>('synthetic');
  const [workflow, setWorkflow] = useState<Workflow>('nr_sld');
  const [nrSldMode, setNrSldMode] = useState<NrSldMode>('train');
  const [autoGenerateModelStats, setAutoGenerateModelStats] = useState(true);

  const [isGenerating, setIsGenerating] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [graphData, setGraphData] = useState<GenerateResponse | null>(null);
  const [consoleLogs, setConsoleLogs] = useState<string[]>([]);
  const [generationStart, setGenerationStart] = useState<number | null>(null);
  const [epochProgress, setEpochProgress] = useState<{ current: number; total: number } | null>(
    null
  );
  const [activeGenerationName, setActiveGenerationName] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<BackendStatus | null>(null);
  const [limits, setLimits] = useState<Limits>(DEFAULT_LIMITS);
  const [isProduction, setIsProduction] = useState(false);
  const [isHydrated, setIsHydrated] = useState(false);

  const [showExplore, setShowExplore] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const [importNamePopup, setImportNamePopup] = useState(false);
  const [pendingImport, setPendingImport] = useState<PendingImport | null>(null);
  const [importName, setImportName] = useState('');

  const addLog = useCallback((message: string) => {
    setGenerationStart((prev) => prev ?? Date.now());
    const timestamp = new Date().toLocaleTimeString();
    setConsoleLogs((prev) => [...prev, `[${timestamp}] ${message}`]);
  }, []);

  const currentParams: BundleParams = {
    layers: filmLayers,
    generator: generatorParams,
    training: trainingParams,
  };

  const handleLoadSaveCore = useCallback(
    (params: BundleParams, result: GenerateResponse) => {
      setFilmLayers(params.layers);
      setGeneratorParams(params.generator);
      setTrainingParams(params.training);
      setGraphData(result);
      addLog('Loaded saved session from history');
    },
    [addLog]
  );

  const downloads = useDownloadBundle({
    apiUrl: API_URL,
    addLog,
    currentParams,
    graphData,
    userId: sessionUserId,
    onLoadSave: handleLoadSaveCore,
  });

	  const {
	    showBundleConfirm,
	    bundlePayload,
	    isDownloadingBundle,
	    bundleSelection,
	    setBundleSelection,
	    bundleEstimate,
	    bundleEstimateError,
	    closeBundleConfirm,
    handleConfirmBundleDownload,
    handleDownloadBundleClick,
    handleExportAll,
    handleSidebarDownloadRequest,
    requestAutoCapture,
    resetPngs,
    setExportPngs,
  } = downloads;

  const saveToHistory = useCallback(
    async (data: { params: BundleParams; result: GenerateResponse; name?: string }) => {
      if (!sessionUserId) return;
      try {
        const { export_pngs: _exportPngs, ...resultRest } = data.result;
        void _exportPngs;
        const payload = {
          layers: data.params.layers,
          generator: data.params.generator,
          training: data.params.training,
          result: { ...resultRest, name: data.name },
          name: data.name,
        };

        const res = await fetch(`${API_URL}/api/history`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-User-ID': sessionUserId,
          },
          body: JSON.stringify(payload),
        });

        if (!res.ok) throw new Error('Failed to save');
        addLog('Saved import to history');
      } catch {
        addLog('Warning: Failed to save import to history');
      }
    },
    [sessionUserId, addLog]
  );

  const confirmImportName = useCallback(() => {
    if (!pendingImport) return;

    const updatedResult: GenerateResponse = {
      ...pendingImport.result,
      name: importName || undefined,
    };
    setGraphData(updatedResult);
    setExportPngs(pendingImport.exportPngs);
    addLog(`Imported data (Name: ${importName || 'Untitled'})`);

    saveToHistory({
      params: pendingImport.params,
      result: updatedResult,
      name: importName || undefined,
    });

    setImportNamePopup(false);
    setPendingImport(null);
    setImportName('');
  }, [pendingImport, importName, addLog, saveToHistory, setExportPngs]);

  const cancelImportName = useCallback(() => {
    setImportNamePopup(false);
    setPendingImport(null);
    setImportName('');
    addLog('Import cancelled');
  }, [addLog]);

  const handleReset = useCallback(() => {
    setFilmLayers(DEFAULT_LAYERS);
    setGeneratorParams(DEFAULT_GENERATOR);
    setTrainingParams(DEFAULT_TRAINING);
    setGraphData(null);
    closeBundleConfirm();
    resetPngs();
    setConsoleLogs([]);
    setGenerationStart(null);
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
  }, [closeBundleConfirm, resetPngs]);

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
      // ignore parse errors
    }
    setIsHydrated(true);
  }, []);

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
    if (!graphData) {
      localStorage.setItem(`${STORAGE_KEY}_graphData`, JSON.stringify(graphData));
      return;
    }
    const { export_pngs: _exportPngs, ...rest } = graphData;
    void _exportPngs;
    localStorage.setItem(`${STORAGE_KEY}_graphData`, JSON.stringify(rest));
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
          addLog(
            `Backend connected. pyreflect: ${status.pyreflect_available ? 'available' : 'unavailable'}`
          );
          if (status.data_files.length > 0) {
            addLog(`Data files: ${status.data_files.join(', ')}`);
          }
        }
        if (limitsRes.ok) {
          const limitsData: LimitsResponse = await limitsRes.json();
          setLimits(limitsData.limits);
          setIsProduction(limitsData.production);
          if (limitsData.production) {
            addLog(
              `Production mode: limits enforced (max ${limitsData.limits.max_curves} curves, ${limitsData.limits.max_epochs} epochs)`
            );
          }
        }
      } catch {
        addLog('Backend not reachable. Start with: python -m uvicorn main:app --reload');
      }
    };
    fetchStatus();
  }, [isHydrated, addLog]);

  const handleGenerate = useCallback(
    async (name?: string) => {
      resetPngs();
      addLog('Starting generation...');

      if (dataSource === 'real') {
        const modeLabel = workflow === 'sld_chi' ? '' : ` (${nrSldMode})`;
        addLog(`Real data: ${workflow}${modeLabel}`);
      }
      if (name) addLog(`Name: ${name}`);
      if (dataSource === 'synthetic') {
        addLog(`Params: ${generatorParams.numCurves} curves, ${trainingParams.epochs} epochs`);
      }

      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (sessionUserId) headers['X-User-ID'] = sessionUserId;

      const payload = {
        layers: filmLayers,
        generator: generatorParams,
        training: trainingParams,
        name,
        dataSource,
        workflow,
        mode: nrSldMode,
        autoGenerateModelStats,
      };

      // Try queue first (fire-and-forget for instant button return)
      fetch(`${API_URL}/api/jobs/submit`, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
      })
        .then(async (queueRes) => {
          if (queueRes.ok) {
            const queueData = await queueRes.json();
            addLog(`Job queued: ${queueData.job_id.slice(0, 8)}... (position: ${queueData.queue_position})`);
            addLog('Job will run in background. Check history when complete.');
          } else {
            // Queue not available - user should use streaming mode manually
            addLog('Queue not available. Click again to use streaming mode.');
          }
        })
        .catch(() => {
          addLog('Queue not reachable. Click again to use streaming mode.');
        });

      // Fire-and-forget - return immediately so user can submit more jobs
    },
    [
      addLog,
      autoGenerateModelStats,
      dataSource,
      resetPngs,
      filmLayers,
      generatorParams,
      nrSldMode,
      sessionUserId,
      trainingParams,
      workflow,
    ]
  );

  const handleUploadFiles = useCallback(
    async (uploads: { file: File; role: UploadRole }[]) => {
      if (uploads.length === 0) return;
      setIsUploading(true);
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

        if (!response.ok) throw new Error(`Upload failed: ${response.statusText}`);

        const result = await response.json();
        const savedCount = Array.isArray(result.saved) ? result.saved.length : 0;
        addLog(`Uploaded ${savedCount} file(s) to backend data folder`);
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'Unknown error';
        addLog(`Upload error: ${errorMsg}`);
      } finally {
        setIsUploading(false);
      }
    },
    [addLog]
  );

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

        let resultData: GenerateResponse;
        let paramsData: BundleParams = {
          layers: DEFAULT_LAYERS,
          generator: DEFAULT_GENERATOR,
          training: DEFAULT_TRAINING,
        };

        if (json?.result && json?.params) {
          resultData = json.result as GenerateResponse;
          paramsData = json.params as BundleParams;
        } else if (json?.nr && json?.sld) {
          resultData = json as GenerateResponse;
        } else {
          throw new Error('Unrecognized JSON format');
        }

        setFilmLayers(paramsData.layers);
        setGeneratorParams(paramsData.generator);
        setTrainingParams(paramsData.training);

        const { export_pngs: _exportPngs, ...resultRest } = resultData;
        const cleanedResult = resultRest as GenerateResponse;
        const importedPngs = _exportPngs ?? null;

        const finalName = cleanedResult.name;
        if (finalName) {
          setGraphData(cleanedResult);
          setExportPngs(importedPngs);
          addLog(`Imported data from ${file.name} (Name: ${finalName})`);
          saveToHistory({ params: paramsData, result: cleanedResult, name: finalName });
        } else {
          setPendingImport({ params: paramsData, result: cleanedResult, exportPngs: importedPngs });
          setImportName('');
          setImportNamePopup(true);
        }
      } catch (err) {
        addLog(`Error parsing JSON: ${err instanceof Error ? err.message : 'Unknown error'}`);
      }
    };
    input.click();
  }, [addLog, setExportPngs, saveToHistory]);

  return (
    <div className="container">
      <AppHeader
        appVersion={APP_VERSION}
        isProduction={isProduction}
        isGenerating={isGenerating}
        epochProgress={epochProgress}
        hasGraphData={Boolean(graphData)}
        session={session ?? null}
        onOpenHistory={() => setShowExplore(true)}
        onImportJson={handleImportJSON}
        onExportJson={handleExportAll}
        onDownloadBundle={handleDownloadBundleClick}
      />

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
            onToggleCollapse={() => setSidebarCollapsed((prev) => !prev)}
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
        userId={sessionUserId}
        onResetLocal={handleReset}
        onLoadSave={(params, result) => {
          resetPngs();
          handleLoadSaveCore(params, result);
        }}
        onRequestDownload={handleSidebarDownloadRequest}
        inProgress={isGenerating ? { name: activeGenerationName, epochProgress } : null}
      />

	      <DownloadBundleModal
	        isOpen={showBundleConfirm}
	        hasPayload={Boolean(bundlePayload)}
	        hasModel={Boolean(bundlePayload?.result.model_id)}
	        isDownloading={isDownloadingBundle}
	        selection={bundleSelection}
	        estimate={bundleEstimate}
	        estimateError={bundleEstimateError}
	        onClose={closeBundleConfirm}
	        onConfirm={handleConfirmBundleDownload}
	        onSelectionChange={(patch) =>
	          setBundleSelection((prev) => ({ ...prev, ...patch }))
	        }
	      />

	      {isDownloadingBundle && (
	        <>
	          <div className="model-download-overlay" />
	          <div className="model-download-popup" aria-live="polite">
	            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', lineHeight: '1.4' }}>
	              <div style={{ fontWeight: 600, marginBottom: '8px' }}>Downloading…</div>
	              <div style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
	                Building bundle and capturing charts. This can take a few seconds.
	              </div>
	            </div>
	          </div>
	        </>
	      )}

	      <ImportNameModal
	        isOpen={importNamePopup}
	        value={importName}
	        onChange={setImportName}
        onCancel={cancelImportName}
        onConfirm={confirmImportName}
      />
    </div>
  );
}
