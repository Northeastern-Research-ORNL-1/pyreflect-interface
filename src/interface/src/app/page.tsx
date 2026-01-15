'use client';

import { useState, useCallback, useEffect } from 'react';
import ParameterPanel from '../components/ParameterPanel';
import GraphDisplay from '../components/GraphDisplay';
import ConsoleOutput from '../components/ConsoleOutput';
import { FilmLayer, GeneratorParams, TrainingParams, GenerateResponse } from '@/types';

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
  numCurves: 50000,
  numFilmLayers: 5,
};

const DEFAULT_TRAINING: TrainingParams = {
  batchSize: 32,
  epochs: 3,
  layers: 12,
  dropout: 0.0,
  latentDim: 16,
  aeEpochs: 50,
  mlpEpochs: 50,
};

export default function Home() {
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
  const [isHydrated, setIsHydrated] = useState(false);

  const addLog = useCallback((message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setConsoleLogs(prev => [...prev, `[${timestamp}] ${message}`]);
  }, []);

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

  // Fetch backend status on mount
  useEffect(() => {
    if (!isHydrated) return;
    const fetchStatus = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/status');
        if (response.ok) {
          const status: BackendStatus = await response.json();
          setBackendStatus(status);
          addLog(`Backend connected. pyreflect: ${status.pyreflect_available ? 'available' : 'unavailable'}`);
          if (status.data_files.length > 0) {
            addLog(`Data files: ${status.data_files.join(', ')}`);
          }
        }
      } catch {
        addLog('Backend not reachable. Start with: python -m uvicorn main:app --reload');
      }
    };
    fetchStatus();
  }, [isHydrated, addLog]);

  const handleGenerate = useCallback(async () => {
    setIsGenerating(true);
    setGenerationStart(Date.now());
    setError(null);
    addLog('Starting generation...');
    addLog(`Params: ${generatorParams.numCurves} curves, ${trainingParams.epochs} epochs`);
    
    try {
      const response = await fetch('http://localhost:8000/api/generate/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          layers: filmLayers,
          generator: generatorParams,
          training: trainingParams,
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
  }, [filmLayers, generatorParams, trainingParams, addLog]);

  const handleUploadFiles = useCallback(async (files: File[]) => {
    if (files.length === 0) return;
    setIsUploading(true);
    setError(null);
    addLog(`Uploading ${files.length} file(s)...`);

    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));

    try {
      const response = await fetch('http://localhost:8000/api/upload', {
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

  const handleExportAll = useCallback(() => {
    if (!graphData) return;
    const json = JSON.stringify(graphData, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'pyreflect_results.json';
    a.click();
    URL.revokeObjectURL(url);
    addLog('Exported all data to pyreflect_results.json');
  }, [graphData, addLog]);

  return (
    <div className="container">
      <header className="header">
        <div className="header__logo">
          <span>◇</span>
          <span>PYREFLECT</span>
        </div>
        <nav className="header__nav">
          {graphData && (
            <button className="header__export-btn" onClick={handleExportAll}>
              ↓ Export All
            </button>
          )}
          <span className={`status ${isGenerating ? 'status--training' : 'status--active'}`}>
            <span className="status__dot"></span>
            {isGenerating ? 'Training...' : 'Ready'}
          </span>
        </nav>
      </header>

      <main className="main">
        <aside className="sidebar">
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
            isUploading={isUploading}
            backendStatus={backendStatus}
          />
        </aside>

        <section className="content">
          <GraphDisplay data={graphData} />
          <ConsoleOutput logs={consoleLogs} isGenerating={isGenerating} startTimeMs={generationStart} />
        </section>
      </main>
    </div>
  );
}

