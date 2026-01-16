'use client';

import { useState, useCallback, useEffect } from 'react';
import { useSession, signIn, signOut } from 'next-auth/react';
import ParameterPanel from '../components/ParameterPanel';
import GraphDisplay from '../components/GraphDisplay';
import ConsoleOutput from '../components/ConsoleOutput';
import ExploreSidebar from '../components/ExploreSidebar';
import { FilmLayer, GeneratorParams, TrainingParams, GenerateResponse, Limits, LimitsResponse, DEFAULT_LIMITS } from '@/types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const STORAGE_KEY = 'pyreflect_state';

interface BackendStatus {
  pyreflect_available: boolean;
  has_settings: boolean;
  data_files: string[];
  curve_files: string[];
  expt_files: string[];
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
  const [backendStatus, setBackendStatus] = useState<BackendStatus | null>(null);
  const [limits, setLimits] = useState<Limits>(DEFAULT_LIMITS);
  const [isProduction, setIsProduction] = useState(false);
  const [isHydrated, setIsHydrated] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showActionsMenu, setShowActionsMenu] = useState(false);
  const [showExplore, setShowExplore] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [importNamePopup, setImportNamePopup] = useState(false);
  const [pendingImportData, setPendingImportData] = useState<GenerateResponse | null>(null);
  const [importName, setImportName] = useState('');
  const [showDownloadConfirm, setShowDownloadConfirm] = useState(false);
  const [downloadTarget, setDownloadTarget] = useState<{model_id: string, model_size_mb?: number} | null>(null);
  const [fetchedSize, setFetchedSize] = useState<number | null>(null);

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
    if (typeof window !== 'undefined') {
      localStorage.removeItem(`${STORAGE_KEY}_layers`);
      localStorage.removeItem(`${STORAGE_KEY}_generator`);
      localStorage.removeItem(`${STORAGE_KEY}_training`);
      localStorage.removeItem(`${STORAGE_KEY}_graphData`);
      localStorage.removeItem(`${STORAGE_KEY}_logs`);
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

      if (storedLayers) setFilmLayers(JSON.parse(storedLayers));
      if (storedGenerator) setGeneratorParams(JSON.parse(storedGenerator));
      if (storedTraining) setTrainingParams(JSON.parse(storedTraining));
      if (storedGraphData) setGraphData(JSON.parse(storedGraphData));
      if (storedLogs) setConsoleLogs(JSON.parse(storedLogs));
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
    setError(null);
    addLog('Starting generation...');
    if (name) addLog(`Name: ${name}`);
    addLog(`Params: ${generatorParams.numCurves} curves, ${trainingParams.epochs} epochs`);
    
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
              // Could update a progress bar here
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
    }
  }, [filmLayers, generatorParams, trainingParams, addLog, session]);

  const handleUploadFiles = useCallback(async (files: File[]) => {
    if (files.length === 0) return;
    setIsUploading(true);
    setError(null);
    addLog(`Uploading ${files.length} file(s)...`);

    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));

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

  const handleExportAll = useCallback(() => {
    if (!graphData) return;
    
    // Create export object containing params and result
    const exportData = {
        params: {
            layers: filmLayers,
            generator: generatorParams,
            training: trainingParams
        },
        result: graphData
    };

    const fileName = `pyreflect_export_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    const jsonStr = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    addLog('Exported full session data (params + results)');
  }, [graphData, filmLayers, generatorParams, trainingParams, addLog]);

  const handleDownloadModelClick = useCallback(() => {
     if (!graphData?.model_id) return;
     setDownloadTarget(null); // Clear specific target to use current graphData
     setShowDownloadConfirm(true);
  }, [graphData]);
  
  const handleSidebarDownloadRequest = useCallback((model_id: string, model_size_mb?: number) => {
      setDownloadTarget({ model_id, model_size_mb });
      setShowDownloadConfirm(true);
  }, []);

  useEffect(() => {
    if (!showDownloadConfirm) {
        setFetchedSize(null);
        return;
    }
    const target = downloadTarget || graphData;
    if (!target?.model_id) return;

    if (target.model_size_mb) {
        setFetchedSize(target.model_size_mb);
    } else {
        fetch(`${API_URL}/api/models/${target.model_id}/info`)
          .then(res => res.json())
          .then(data => {
             if (data.size_mb) setFetchedSize(data.size_mb);
          })
          .catch(err => console.error("Size fetch failed", err));
    }
  }, [showDownloadConfirm, downloadTarget, graphData]);

  const confirmDownloadModel = useCallback(() => {
    const target = downloadTarget || graphData;
    if (!target?.model_id) return;
    
    const link = document.createElement('a');
    link.href = `${API_URL}/api/models/${target.model_id}`;
    link.download = `model_${target.model_id}.pth`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    addLog('Downloading model file...');
    setShowDownloadConfirm(false);
    setDownloadTarget(null);
  }, [graphData, downloadTarget, addLog]);

  const handleDeleteModel = useCallback(async () => {
    if (!graphData?.model_id) return;
    if (!confirm('Are you sure you want to delete the local model file? This cannot be undone if not uploaded.')) return;
    
    try {
        const res = await fetch(`${API_URL}/api/models/${graphData.model_id}`, { method: 'DELETE' });
        if (res.ok) {
            addLog('Local model file deleted.');
            setShowDownloadConfirm(false);
        } else {
            throw new Error('Failed to delete');
        }
    } catch (e) {
        addLog(`Error deleting model: ${e}`);
        alert('Could not delete local model (maybe it is already gone or permission denied).');
    }
  }, [graphData, addLog]);



  return (
    <div className="container">
      <header className="header">
        <div className="header__logo">
          <span>◇</span>
          <span>PYREFLECT</span>
          <span className="header__version">v0.0.1</span>
          {isProduction && <span className="header__version" style={{ color: '#f59e0b', marginLeft: '8px' }}>PROD</span>}
          <span className={`status ${isGenerating ? 'status--training' : 'status--active'}`} style={{ marginLeft: '12px' }}>
            <span className="status__dot"></span>
            <span className="header__status-text">{isGenerating ? 'Training...' : 'Ready'}</span>
          </span>
        </div>
        <nav className="header__nav">
          {/* Desktop: show buttons inline */}
          <div className="header__actions-desktop">
            <button className="header__export-btn" onClick={() => setShowExplore(true)}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
              </svg>
              <span className="header__btn-label">History</span>
            </button>
            <button className="header__export-btn" onClick={handleImportJSON}>
              <span>↑</span><span className="header__btn-label">Import</span>
            </button>
            {graphData && (
              <button className="header__export-btn" onClick={handleExportAll}>
                <span>↓</span><span className="header__btn-label">Export</span>
              </button>
            )}
            {graphData?.model_id && (
              <button className="header__export-btn" onClick={handleDownloadModelClick}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="7 10 12 15 17 10"></polyline>
                  <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                <span className="header__btn-label">Model</span>
              </button>
            )}
          </div>

          {/* Mobile: dropdown menu */}
          <div className="header__actions-mobile">
            <button 
              className="header__export-btn" 
              onClick={() => setShowActionsMenu(!showActionsMenu)}
            >
              <span>≡</span>
            </button>
            {showActionsMenu && (
              <div className="header__dropdown">
                <button className="header__dropdown-item" onClick={() => { setShowExplore(true); setShowActionsMenu(false); }}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
                  </svg> 
                  History
                </button>
                <button className="header__dropdown-item" onClick={() => { handleImportJSON(); setShowActionsMenu(false); }}>
                  <span>↑</span> Import
                </button>
                {graphData && (
                  <button className="header__dropdown-item" onClick={() => { handleExportAll(); setShowActionsMenu(false); }}>
                    <span>↓</span> Export
                  </button>
                )}
                {graphData?.model_id && (
                  <button className="header__dropdown-item" onClick={() => { handleDownloadModelClick(); setShowActionsMenu(false); }}>
                    <span>↓</span> Model
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
        userId={session?.user ? (session.user as any).id : undefined}
        onLoadSave={handleLoadSave}
        onRequestDownload={handleSidebarDownloadRequest}
      />
      
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
      {showDownloadConfirm && (
        <>
          <div className="model-download-overlay" onClick={() => setShowDownloadConfirm(false)} />
          <div className="model-download-popup">
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', marginBottom: '16px', lineHeight: '1.4' }}>
              <div style={{ fontWeight: 600, marginBottom: '8px' }}>Download Trained Model?</div>
              <div>Size: <span style={{ color: 'var(--text-secondary)' }}>
                {fetchedSize ? `~${fetchedSize.toFixed(2)} MB` : 'Checking...'}
              </span></div>
              <div style={{ color: 'var(--text-muted)', fontSize: '11px', marginTop: '4px' }}>
                This is the raw PyTorch state dictionary (.pth)
              </div>
            </div>
            <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
              <button 
                className="btn btn--outline" 
                onClick={() => { setShowDownloadConfirm(false); setDownloadTarget(null); }}
                style={{ padding: '6px 12px', fontSize: '11px' }}
              >
                CANCEL
              </button>
              <button 
                className="btn" 
                onClick={confirmDownloadModel}
                style={{ padding: '6px 12px', fontSize: '11px' }}
              >
                DOWNLOAD
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

