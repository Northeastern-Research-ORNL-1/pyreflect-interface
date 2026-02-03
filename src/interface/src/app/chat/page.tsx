'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import WelcomeScreen from './components/WelcomeScreen';
import Message from './components/Message';
import ChatInput from './components/ChatInput';
import GraphDisplay from '@/components/GraphDisplay';
import { GenerateResponse } from '@/types';

interface MessageType {
  role: 'user' | 'assistant';
  content: string;
}

interface LayerConfig {
  name: string;
  sld: number;
  isld: number;
  thickness: number;
  roughness: number;
}

interface GenerationConfig {
  substrate: string;
  layers: { name: string; thickness: number; sld: number; roughness: number }[];
  environment: string;
  numCurves: number;
  epochs: number;
}

interface HistoryItem {
  id: string;
  config: GenerationConfig;
  result: GenerateResponse;
  timestamp: Date;
  duration: number;
}

const SLD_VALUES: Record<string, number> = {
  'silicon': 2.07, 'si': 2.07,
  'sio2': 3.47, 'silicon dioxide': 3.47, 'silica': 3.47,
  'air': 0,
  'd2o': 6.36, 'heavy water': 6.36,
  'h2o': -0.56, 'water': -0.56,
  'gold': 4.5, 'au': 4.5,
  'titanium': -1.95, 'ti': -1.95,
  'pmma': 1.0, 'polystyrene': 1.04, 'ps': 1.04,
};

const TEST_CONFIGS: GenerationConfig[] = [
  {
    substrate: 'silicon',
    layers: [
      { name: 'SiO2', thickness: 15, sld: 3.47, roughness: 3 },
      { name: 'PMMA', thickness: 100, sld: 1.0, roughness: 5 },
    ],
    environment: 'air',
    numCurves: 100,
    epochs: 10
  },
  {
    substrate: 'silicon',
    layers: [
      { name: 'Gold', thickness: 50, sld: 4.5, roughness: 2 },
      { name: 'Polymer', thickness: 150, sld: 1.2, roughness: 8 },
    ],
    environment: 'd2o',
    numCurves: 100,
    epochs: 10
  }
];

function getRandomTestConfig(): GenerationConfig {
  return TEST_CONFIGS[Math.floor(Math.random() * TEST_CONFIGS.length)];
}

function getSLD(material: string): number {
  const key = material.toLowerCase().trim();
  return SLD_VALUES[key] ?? 1.0;
}

function buildLayersPayload(config: GenerationConfig): LayerConfig[] {
  const result: LayerConfig[] = [];
  result.push({ name: config.substrate.toLowerCase(), sld: getSLD(config.substrate), isld: 0.0, thickness: 0.0, roughness: 1.8 });
  if (config.substrate.toLowerCase().includes('silicon')) {
    result.push({ name: 'siox', sld: 3.47, isld: 0.0, thickness: 12.0, roughness: 2.0 });
  }
  config.layers.forEach((layer, i) => {
    result.push({ name: layer.name?.toLowerCase() || `layer_${i + 1}`, sld: layer.sld ?? getSLD(layer.name || ''), isld: 0.0, thickness: layer.thickness ?? 100.0, roughness: layer.roughness ?? 5.0 });
  });
  result.push({ name: config.environment.toLowerCase(), sld: getSLD(config.environment), isld: 0.0, thickness: 0.0, roughness: 0.0 });
  return result;
}

async function callGenerateAPI(config: GenerationConfig): Promise<GenerateResponse> {
  const layers = buildLayersPayload(config);
  const response = await fetch('http://127.0.0.1:8000/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      layers,
      generator: { numCurves: config.numCurves, numFilmLayers: layers.length - 2 },
      training: { batchSize: 32, epochs: config.epochs, layers: 12, dropout: 0.0, latentDim: 16, aeEpochs: 50, mlpEpochs: 50 }
    })
  });
  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || 'Generation failed');
  }
  return response.json();
}

function extractJSON(text: string): any | null {
  const fenceMatch = text.match(/```json\s*([\s\S]*?)\s*```/);
  if (fenceMatch) { try { return JSON.parse(fenceMatch[1]); } catch { return null; } }
  const rawMatch = text.match(/\{\s*"ready_to_generate"\s*:\s*true[\s\S]*\}/);
  if (rawMatch) { try { return JSON.parse(rawMatch[0]); } catch { return null; } }
  return null;
}

function isTestCommand(text: string): boolean {
  const lower = text.toLowerCase().trim();
  return ['test', 'quick test', 'demo', 'sample', 'random'].some(p => lower.includes(p));
}

function formatDuration(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  const tenths = Math.floor((ms % 1000) / 100);
  if (min > 0) return `${min}:${sec.toString().padStart(2, '0')}.${tenths}`;
  return `${sec}.${tenths}s`;
}

// Timer Component
function LiveTimer({ startTime, isRunning }: { startTime: number | null; isRunning: boolean }) {
  const [elapsed, setElapsed] = useState(0);
  
  useEffect(() => {
    if (!isRunning || !startTime) return;
    const interval = setInterval(() => setElapsed(Date.now() - startTime), 100);
    return () => clearInterval(interval);
  }, [isRunning, startTime]);

  if (!isRunning) return null;

  return (
    <div style={{
      position: 'absolute',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      textAlign: 'center',
      zIndex: 10
    }}>
      <div style={{
        fontSize: '64px',
        fontWeight: 200,
        fontFamily: "'JetBrains Mono', monospace",
        color: '#10b981',
        letterSpacing: '-0.02em',
        textShadow: '0 0 40px rgba(16, 185, 129, 0.3)'
      }}>
        {formatDuration(elapsed)}
      </div>
      <div style={{ fontSize: '12px', color: '#666', textTransform: 'uppercase', letterSpacing: '0.1em', marginTop: '8px' }}>
        Generating curves...
      </div>
      <div style={{ marginTop: '24px' }}>
        <div style={{ width: '200px', height: '2px', background: '#1a1a1a', borderRadius: '1px', overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            background: 'linear-gradient(90deg, #10b981, #059669)',
            animation: 'pulse 1.5s ease-in-out infinite',
            width: '40%'
          }} />
        </div>
      </div>
      <style>{`@keyframes pulse { 0%, 100% { transform: translateX(-100%); } 50% { transform: translateX(250%); } }`}</style>
    </div>
  );
}

// Collapsible Parameter Panel
function ParameterPanel({ 
  config, onChange, onGenerate, isGenerating, isCollapsed, onToggle 
}: { 
  config: GenerationConfig | null;
  onChange: (c: GenerationConfig) => void;
  onGenerate: () => void;
  isGenerating: boolean;
  isCollapsed: boolean;
  onToggle: () => void;
}) {
  return (
    <div style={{
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      background: '#0d0d0d',
      borderBottom: '1px solid #2a2a2a',
      transform: isCollapsed ? 'translateY(-100%)' : 'translateY(0)',
      transition: 'transform 0.3s ease',
      zIndex: 20,
      maxHeight: '45vh',
      overflow: 'auto'
    }}>
      <div style={{ padding: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.1em', color: '#888' }}>
            Parameters
          </span>
          {config && <span style={{ fontSize: '10px', color: '#10b981' }}>● Ready</span>}
        </div>
        
        {config ? (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '16px' }}>
              <div>
                <label style={{ fontSize: '10px', color: '#666', textTransform: 'uppercase' }}>Substrate</label>
                <input value={config.substrate} onChange={e => onChange({ ...config, substrate: e.target.value })}
                  style={{ width: '100%', padding: '10px', background: '#1a1a1a', border: '1px solid #333', color: 'white', fontFamily: 'monospace', fontSize: '12px', marginTop: '4px' }} />
              </div>
              <div>
                <label style={{ fontSize: '10px', color: '#666', textTransform: 'uppercase' }}>Environment</label>
                <select value={config.environment} onChange={e => onChange({ ...config, environment: e.target.value })}
                  style={{ width: '100%', padding: '10px', background: '#1a1a1a', border: '1px solid #333', color: 'white', fontFamily: 'monospace', fontSize: '12px', marginTop: '4px' }}>
                  <option value="air">Air</option>
                  <option value="d2o">D₂O</option>
                  <option value="h2o">H₂O</option>
                </select>
              </div>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <label style={{ fontSize: '10px', color: '#666', textTransform: 'uppercase' }}>Layers ({config.layers.length})</label>
              <div style={{ marginTop: '8px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {config.layers.map((layer, i) => (
                  <div key={i} style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr', gap: '8px', padding: '10px', background: '#1a1a1a', border: '1px solid #252525' }}>
                    <input placeholder="Name" value={layer.name} onChange={e => { const l = [...config.layers]; l[i] = { ...layer, name: e.target.value }; onChange({ ...config, layers: l }); }}
                      style={{ padding: '6px', background: '#0a0a0a', border: '1px solid #333', color: 'white', fontFamily: 'monospace', fontSize: '11px' }} />
                    <input type="number" placeholder="Å" value={layer.thickness} onChange={e => { const l = [...config.layers]; l[i] = { ...layer, thickness: +e.target.value }; onChange({ ...config, layers: l }); }}
                      style={{ padding: '6px', background: '#0a0a0a', border: '1px solid #333', color: 'white', fontFamily: 'monospace', fontSize: '11px' }} />
                    <input type="number" step="0.1" placeholder="SLD" value={layer.sld} onChange={e => { const l = [...config.layers]; l[i] = { ...layer, sld: +e.target.value }; onChange({ ...config, layers: l }); }}
                      style={{ padding: '6px', background: '#0a0a0a', border: '1px solid #333', color: 'white', fontFamily: 'monospace', fontSize: '11px' }} />
                    <input type="number" placeholder="σ" value={layer.roughness} onChange={e => { const l = [...config.layers]; l[i] = { ...layer, roughness: +e.target.value }; onChange({ ...config, layers: l }); }}
                      style={{ padding: '6px', background: '#0a0a0a', border: '1px solid #333', color: 'white', fontFamily: 'monospace', fontSize: '11px' }} />
                  </div>
                ))}
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 2fr', gap: '12px' }}>
              <div>
                <label style={{ fontSize: '10px', color: '#666', textTransform: 'uppercase' }}>Curves</label>
                <input type="number" value={config.numCurves} onChange={e => onChange({ ...config, numCurves: +e.target.value })}
                  style={{ width: '100%', padding: '10px', background: '#1a1a1a', border: '1px solid #333', color: 'white', fontFamily: 'monospace', fontSize: '12px', marginTop: '4px' }} />
              </div>
              <div>
                <label style={{ fontSize: '10px', color: '#666', textTransform: 'uppercase' }}>Epochs</label>
                <input type="number" value={config.epochs} onChange={e => onChange({ ...config, epochs: +e.target.value })}
                  style={{ width: '100%', padding: '10px', background: '#1a1a1a', border: '1px solid #333', color: 'white', fontFamily: 'monospace', fontSize: '12px', marginTop: '4px' }} />
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-end' }}>
                <button onClick={onGenerate} disabled={isGenerating}
                  style={{ width: '100%', padding: '12px', background: isGenerating ? '#333' : '#10b981', color: isGenerating ? '#666' : 'black', border: 'none', fontFamily: 'monospace', fontSize: '11px', fontWeight: 600, textTransform: 'uppercase', cursor: isGenerating ? 'not-allowed' : 'pointer' }}>
                  {isGenerating ? 'Generating...' : 'Generate'}
                </button>
              </div>
            </div>
          </>
        ) : (
          <div style={{ color: '#666', fontSize: '12px', textAlign: 'center', padding: '20px' }}>
            Chat with AI or click Quick Test to configure
          </div>
        )}
      </div>

      {/* Toggle Button - Always visible at bottom */}
      <button onClick={onToggle}
        style={{ position: 'absolute', bottom: '-32px', left: '50%', transform: 'translateX(-50%)', background: '#1a1a1a', border: '1px solid #333', borderTop: 'none', color: '#888', padding: '4px 16px', cursor: 'pointer', fontSize: '10px', fontFamily: 'monospace', display: 'flex', alignItems: 'center', gap: '6px' }}>
        <span style={{ transform: isCollapsed ? 'rotate(180deg)' : 'rotate(0)', transition: 'transform 0.3s' }}>▲</span>
        {isCollapsed ? 'SHOW PARAMS' : 'HIDE PARAMS'}
      </button>
    </div>
  );
}

// History Panel
function HistoryPanel({ history, isOpen, onClose, onSelect }: { history: HistoryItem[]; isOpen: boolean; onClose: () => void; onSelect: (item: HistoryItem) => void; }) {
  return (
    <>
      {isOpen && <div onClick={onClose} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 40 }} />}
      <div style={{
        position: 'fixed',
        top: 0,
        right: 0,
        bottom: 0,
        width: '320px',
        background: '#0d0d0d',
        borderLeft: '1px solid #2a2a2a',
        transform: isOpen ? 'translateX(0)' : 'translateX(100%)',
        transition: 'transform 0.3s ease',
        zIndex: 50,
        display: 'flex',
        flexDirection: 'column'
      }}>
        <div style={{ padding: '16px', borderBottom: '1px solid #2a2a2a', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Generation History</span>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer', fontSize: '18px' }}>×</button>
        </div>
        <div style={{ flex: 1, overflow: 'auto', padding: '12px' }}>
          {history.length === 0 ? (
            <div style={{ color: '#666', fontSize: '12px', textAlign: 'center', padding: '40px 20px' }}>
              No generations yet.<br />Results will appear here.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {history.map((item, i) => (
                <button key={item.id} onClick={() => onSelect(item)}
                  style={{ background: '#1a1a1a', border: '1px solid #333', padding: '12px', cursor: 'pointer', textAlign: 'left', transition: 'border-color 0.2s' }}
                  onMouseOver={e => (e.currentTarget.style.borderColor = '#10b981')}
                  onMouseOut={e => (e.currentTarget.style.borderColor = '#333')}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                    <span style={{ fontSize: '11px', color: '#10b981', fontWeight: 600 }}>Run #{history.length - i}</span>
                    <span style={{ fontSize: '10px', color: '#666' }}>{formatDuration(item.duration)}</span>
                  </div>
                  <div style={{ fontSize: '11px', color: '#ccc', marginBottom: '4px' }}>
                    {item.config.layers.map(l => l.name).join(' → ')}
                  </div>
                  <div style={{ fontSize: '10px', color: '#666' }}>
                    R² {item.result.metrics.r2.toFixed(3)} · MSE {item.result.metrics.mse.toFixed(4)}
                  </div>
                  <div style={{ fontSize: '9px', color: '#444', marginTop: '4px' }}>
                    {item.timestamp.toLocaleTimeString()}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

// Status Bar
function StatusBar({ history, isGenerating, lastDuration, onHistoryClick }: { history: HistoryItem[]; isGenerating: boolean; lastDuration: number | null; onHistoryClick: () => void; }) {
  return (
    <div style={{
      height: '40px',
      borderTop: '1px solid #2a2a2a',
      background: '#0d0d0d',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 16px',
      fontSize: '11px',
      fontFamily: "'JetBrains Mono', monospace"
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <span style={{ color: isGenerating ? '#10b981' : '#666' }}>
          {isGenerating ? '● RUNNING' : '○ IDLE'}
        </span>
        {lastDuration && !isGenerating && (
          <span style={{ color: '#666' }}>Last: {formatDuration(lastDuration)}</span>
        )}
      </div>
      <button onClick={onHistoryClick}
        style={{ background: 'none', border: '1px solid #333', color: '#888', padding: '4px 12px', cursor: 'pointer', fontSize: '10px', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: '6px' }}>
        <span>📊</span> History ({history.length})
      </button>
    </div>
  );
}

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  data: string | ArrayBuffer | null;
  preview?: string[][];
}

export default function ChatPage() {
  const [messages, setMessages] = useState<MessageType[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [graphData, setGraphData] = useState<GenerateResponse | null>(null);
  const [generationStart, setGenerationStart] = useState<number | null>(null);
  const [pendingConfig, setPendingConfig] = useState<GenerationConfig | null>(null);
  const [paramsCollapsed, setParamsCollapsed] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [lastDuration, setLastDuration] = useState<number | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [showFilePreview, setShowFilePreview] = useState<UploadedFile | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    Array.from(files).forEach(file => {
      const reader = new FileReader();
      reader.onload = (event) => {
        const newFile: UploadedFile = {
          id: Math.random().toString(36).substr(2, 9),
          name: file.name,
          size: file.size,
          type: file.type,
          data: event.target?.result || null
        };

        // Parse CSV preview
        if (file.name.endsWith('.csv') || file.type === 'text/csv') {
          const text = event.target?.result as string;
          const lines = text.split('\n').slice(0, 6);
          newFile.preview = lines.map(line => line.split(',').map(cell => cell.trim()));
        }

        setUploadedFiles(prev => [...prev, newFile]);
      };

      if (file.name.endsWith('.csv') || file.type === 'text/csv' || file.type.startsWith('text/')) {
        reader.readAsText(file);
      } else {
        reader.readAsArrayBuffer(file);
      }
    });

    e.target.value = '';
  }, []);

  const removeFile = useCallback((id: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== id));
    if (showFilePreview?.id === id) setShowFilePreview(null);
  }, [showFilePreview]);

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const handleGeneration = useCallback(async (config: GenerationConfig) => {
    setIsGenerating(true);
    setParamsCollapsed(true);
    const start = Date.now();
    setGenerationStart(start);
    
    try {
      const result = await callGenerateAPI(config);
      const duration = Date.now() - start;
      setLastDuration(duration);
      setGraphData(result);
      
      const historyItem: HistoryItem = {
        id: result.model_id || Math.random().toString(36).substr(2, 9),
        config,
        result,
        timestamp: new Date(),
        duration
      };
      setHistory(prev => [historyItem, ...prev]);
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `✅ **Generation complete!** (${formatDuration(duration)})\n\nModel ID: \`${result.model_id}\`\nR²: ${result.metrics.r2.toFixed(4)} · MSE: ${result.metrics.mse.toFixed(4)}`
      }]);
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Unknown error';
      setMessages(prev => [...prev, { role: 'assistant', content: `❌ **Generation failed:** ${msg}` }]);
    } finally {
      setIsGenerating(false);
      setGenerationStart(null);
    }
  }, []);

  const handleQuickTest = useCallback(() => {
    const testConfig = getRandomTestConfig();
    setPendingConfig(testConfig);
    const layerSummary = testConfig.layers.map(l => `${l.name} (${l.thickness}Å)`).join(', ');
    setMessages(prev => [...prev, 
      { role: 'user', content: '🧪 Quick Test' },
      { role: 'assistant', content: `**Test Configuration:**\n- **Substrate:** ${testConfig.substrate}\n- **Layers:** ${layerSummary}\n- **Environment:** ${testConfig.environment}` }
    ]);
    handleGeneration(testConfig);
  }, [handleGeneration]);

  const sendMessage = async (messageText?: string) => {
    const text = messageText || input;
    if (!text.trim() || isLoading) return;
    if (isTestCommand(text)) { setInput(''); handleQuickTest(); return; }

    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setInput('');
    setIsLoading(true);

    try {
      const apiKey = process.env.NEXT_PUBLIC_OPENROUTER_API_KEY;
      if (!apiKey) throw new Error('API key not found');

      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'nousresearch/hermes-3-llama-3.1-405b:free',
          messages: [
            { role: 'system', content: `You are PyReflect AI. Help users set up neutron reflectivity experiments. Ask ONE question at a time. When ready, output JSON with ready_to_generate: true, substrate, layers array, and environment. Common SLDs: Silicon 2.07, SiO2 3.47, Air 0, D2O 6.36, Gold 4.5, PMMA 1.0` },
            ...messages.map(m => ({ role: m.role, content: m.content })),
            { role: 'user', content: text }
          ],
        }),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error?.message || 'Request failed');
      const assistantContent = data.choices[0].message.content;
      setMessages(prev => [...prev, { role: 'assistant', content: assistantContent }]);

      const config = extractJSON(assistantContent);
      if (config?.ready_to_generate) {
        const genConfig: GenerationConfig = { substrate: config.substrate || 'silicon', layers: config.layers || [], environment: config.environment || 'air', numCurves: 100, epochs: 10 };
        setPendingConfig(genConfig);
        await handleGeneration(genConfig);
      }
    } catch (error) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${error instanceof Error ? error.message : 'Unknown'}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setMessages([]);
    setGraphData(null);
    setPendingConfig(null);
    setParamsCollapsed(false);
  };

  const handleHistorySelect = (item: HistoryItem) => {
    setGraphData(item.result);
    setPendingConfig(item.config);
    setHistoryOpen(false);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', backgroundColor: '#0a0a0a', fontFamily: "'JetBrains Mono', 'SF Mono', monospace" }}>
      {/* Header */}
      <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 20px', height: '56px', borderBottom: '1px solid #2a2a2a' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ width: '32px', height: '32px', background: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="black" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
            </svg>
          </div>
          <span style={{ color: 'white', fontWeight: 600, fontSize: '14px' }}>PYREFLECT AI</span>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button onClick={handleQuickTest} disabled={isGenerating || isLoading}
            style={{ color: 'black', background: '#10b981', border: 'none', padding: '6px 12px', cursor: isGenerating ? 'not-allowed' : 'pointer', fontSize: '11px', fontFamily: 'inherit', textTransform: 'uppercase', fontWeight: 600, opacity: isGenerating ? 0.5 : 1 }}>
            🧪 Quick Test
          </button>
          <button onClick={handleReset}
            style={{ color: '#888', background: 'none', border: '1px solid #333', padding: '6px 12px', cursor: 'pointer', fontSize: '11px', fontFamily: 'inherit', textTransform: 'uppercase' }}>
            New Chat
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Left: Chat */}
        <div style={{ width: '45%', display: 'flex', flexDirection: 'column', borderRight: '1px solid #2a2a2a' }}>
          <div style={{ flex: 1, overflow: 'auto', padding: '16px' }}>
            {messages.length === 0 ? <WelcomeScreen onSuggestionClick={sendMessage} /> : (
              <div>
                {messages.map((m, i) => <Message key={i} role={m.role} content={m.content} />)}
                {isLoading && <div style={{ padding: '16px', color: '#666', fontSize: '12px' }}>Thinking...</div>}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
          {/* Chat Input with File Upload */}
          <div style={{ borderTop: '1px solid #2a2a2a', padding: '12px 16px' }}>
            {/* Uploaded Files */}
            {uploadedFiles.length > 0 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '12px' }}>
                {uploadedFiles.map(file => (
                  <div key={file.id}
                    style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 10px', background: '#1a1a1a', border: '1px solid #333', fontSize: '11px' }}>
                    <span style={{ color: file.preview ? '#10b981' : '#888' }}>
                      {file.name.endsWith('.csv') ? '📊' : file.name.endsWith('.txt') ? '📄' : '📁'}
                    </span>
                    <button onClick={() => file.preview && setShowFilePreview(file)}
                      style={{ background: 'none', border: 'none', color: '#ccc', cursor: file.preview ? 'pointer' : 'default', padding: 0, fontFamily: 'inherit', fontSize: '11px' }}>
                      {file.name.length > 20 ? file.name.slice(0, 17) + '...' : file.name}
                    </button>
                    <span style={{ color: '#666', fontSize: '10px' }}>{formatFileSize(file.size)}</span>
                    <button onClick={() => removeFile(file.id)}
                      style={{ background: 'none', border: 'none', color: '#666', cursor: 'pointer', padding: '0 0 0 4px', fontSize: '14px', lineHeight: 1 }}>×</button>
                  </div>
                ))}
              </div>
            )}

            {/* Input Row */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <input type="file" ref={fileInputRef} onChange={handleFileUpload} multiple accept=".csv,.txt,.dat,.json,.xml" style={{ display: 'none' }} />
              <button onClick={() => fileInputRef.current?.click()}
                style={{ width: '36px', height: '36px', background: 'none', border: '1px solid #333', color: '#888', cursor: 'pointer', fontSize: '18px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, transition: 'all 0.2s' }}
                onMouseOver={e => { e.currentTarget.style.borderColor = '#10b981'; e.currentTarget.style.color = '#10b981'; }}
                onMouseOut={e => { e.currentTarget.style.borderColor = '#333'; e.currentTarget.style.color = '#888'; }}>
                +
              </button>
              <input
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendMessage()}
                placeholder={uploadedFiles.length > 0 ? "Describe your data or ask a question..." : "Ask anything"}
                disabled={isLoading || isGenerating}
                style={{ flex: 1, padding: '10px 14px', background: '#1a1a1a', border: '1px solid #333', color: 'white', fontFamily: 'inherit', fontSize: '13px', outline: 'none' }}
              />
              <button onClick={() => sendMessage()} disabled={isLoading || isGenerating || (!input.trim() && uploadedFiles.length === 0)}
                style={{ width: '36px', height: '36px', background: (input.trim() || uploadedFiles.length > 0) ? '#10b981' : '#333', border: 'none', color: (input.trim() || uploadedFiles.length > 0) ? 'black' : '#666', cursor: (input.trim() || uploadedFiles.length > 0) ? 'pointer' : 'not-allowed', fontSize: '14px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                ↑
              </button>
            </div>

            <div style={{ fontSize: '10px', color: '#444', marginTop: '8px', textAlign: 'center' }}>
              Upload .csv, .txt, .dat, .json data files • AI will help analyze your reflectivity data
            </div>
          </div>
        </div>

        {/* Right: Results */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative', overflow: 'hidden' }}>
          {/* Collapsible Parameters */}
          <ParameterPanel config={pendingConfig} onChange={setPendingConfig} onGenerate={() => pendingConfig && handleGeneration(pendingConfig)} isGenerating={isGenerating} isCollapsed={paramsCollapsed} onToggle={() => setParamsCollapsed(!paramsCollapsed)} />

          {/* Graph Area */}
          <div style={{ flex: 1, overflow: 'auto', padding: '16px', paddingTop: paramsCollapsed ? '48px' : '48px', position: 'relative' }}>
            <LiveTimer startTime={generationStart} isRunning={isGenerating} />
            
            {graphData && !isGenerating ? (
              <GraphDisplay data={graphData} />
            ) : !isGenerating && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#444', fontSize: '12px', textAlign: 'center' }}>
                <div>
                  <div style={{ fontSize: '48px', marginBottom: '16px', opacity: 0.3 }}>◇</div>
                  <div>Click <strong>Quick Test</strong> or chat with AI</div>
                </div>
              </div>
            )}
          </div>

          {/* Status Bar */}
          <StatusBar history={history} isGenerating={isGenerating} lastDuration={lastDuration} onHistoryClick={() => setHistoryOpen(true)} />
        </div>
      </div>

      {/* History Panel */}
      <HistoryPanel history={history} isOpen={historyOpen} onClose={() => setHistoryOpen(false)} onSelect={handleHistorySelect} />

      {/* File Preview Modal */}
      {showFilePreview && (
        <>
          <div onClick={() => setShowFilePreview(null)} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 60 }} />
          <div style={{ position: 'fixed', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', background: '#0d0d0d', border: '1px solid #333', padding: '20px', zIndex: 70, maxWidth: '80vw', maxHeight: '70vh', overflow: 'auto', minWidth: '400px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <div>
                <div style={{ fontSize: '14px', fontWeight: 600, color: 'white' }}>{showFilePreview.name}</div>
                <div style={{ fontSize: '11px', color: '#666', marginTop: '2px' }}>{formatFileSize(showFilePreview.size)} • CSV Preview (first 5 rows)</div>
              </div>
              <button onClick={() => setShowFilePreview(null)} style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer', fontSize: '20px' }}>×</button>
            </div>
            {showFilePreview.preview && (
              <div style={{ overflow: 'auto' }}>
                <table style={{ borderCollapse: 'collapse', fontSize: '11px', fontFamily: 'monospace' }}>
                  <thead>
                    <tr>
                      {showFilePreview.preview[0]?.map((cell, i) => (
                        <th key={i} style={{ padding: '8px 12px', background: '#1a1a1a', border: '1px solid #333', color: '#10b981', textAlign: 'left', fontWeight: 600 }}>{cell}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {showFilePreview.preview.slice(1).map((row, i) => (
                      <tr key={i}>
                        {row.map((cell, j) => (
                          <td key={j} style={{ padding: '6px 12px', border: '1px solid #252525', color: '#ccc' }}>{cell}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <div style={{ marginTop: '16px', display: 'flex', gap: '8px' }}>
              <button onClick={() => setShowFilePreview(null)} style={{ flex: 1, padding: '10px', background: '#1a1a1a', border: '1px solid #333', color: '#888', cursor: 'pointer', fontFamily: 'inherit', fontSize: '11px', textTransform: 'uppercase' }}>Close</button>
              <button style={{ flex: 1, padding: '10px', background: '#10b981', border: 'none', color: 'black', cursor: 'pointer', fontFamily: 'inherit', fontSize: '11px', textTransform: 'uppercase', fontWeight: 600 }}>Use This Data</button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}