'use client';
import { pyreflectAPI } from '../../services/pyreflectAPI';
import { useState, useRef, useEffect, useCallback } from 'react';
import WelcomeScreen from './components/WelcomeScreen';
import Message from './components/Message';
import GraphDisplay from '@/components/GraphDisplay';
import { GenerateResponse } from '@/types';

// Enhanced interfaces
interface MessageType {
  role: 'user' | 'assistant';
  content: string;
  suggestions?: SmartSuggestion[];
  isLatest?: boolean; // NEW: track which message is the latest assistant message
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

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  data: string | ArrayBuffer | null;
  preview?: string[][];
}

// NEW: Parameter and Suggestion interfaces
interface CollectedParameter {
  key: string;
  label: string;
  value: string;
  timestamp: Date;
  confidence: 'high' | 'medium' | 'low';
  category: 'material' | 'thickness' | 'environment' | 'substrate' | 'analysis';
}

interface SmartSuggestion {
  text: string;
  category: 'material' | 'thickness' | 'environment' | 'analysis' | 'quick';
  confidence: number;
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

// NEW: Helper functions for parameter extraction
function extractParametersFromMessage(message: string, role: 'user' | 'assistant'): CollectedParameter[] {
  const params: CollectedParameter[] = [];
  const timestamp = new Date();
  const lowerMessage = message.toLowerCase();
  
  // Extract materials
  const materialPatterns = [
    /(?:made of|material|layer.*?is|using)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)/gi,
    /([a-zA-Z]+)\s+(?:film|layer|coating)/gi,
    /(silicon|gold|pmma|polystyrene|titanium|sio2|polymer)\b/gi
  ];
  
  materialPatterns.forEach(pattern => {
    const matches = [...message.matchAll(pattern)];
    matches.forEach(match => {
      const material = match[1]?.trim();
      if (material && material.length > 2 && material.length < 20) {
        params.push({
          key: `material_${material}_${timestamp.getTime()}`,
          label: 'Material',
          value: material,
          timestamp,
          confidence: SLD_VALUES[material.toLowerCase()] ? 'high' : 'medium',
          category: 'material'
        });
      }
    });
  });

  // Extract thickness values
  const thicknessPatterns = [
    /(\d+(?:\.\d+)?)\s*(?:nm|nanometers?|Å|angstroms?)\b/gi,
    /(?:thick(?:ness)?|layer).*?(\d+(?:\.\d+)?)/gi
  ];
  
  thicknessPatterns.forEach(pattern => {
    const matches = [...message.matchAll(pattern)];
    matches.forEach(match => {
      const thickness = match[1];
      if (thickness && parseFloat(thickness) > 0 && parseFloat(thickness) < 10000) {
        params.push({
          key: `thickness_${thickness}_${timestamp.getTime()}`,
          label: 'Thickness',
          value: `${thickness} Å`,
          timestamp,
          confidence: 'high',
          category: 'thickness'
        });
      }
    });
  });

  // Extract environment
  const environments = ['air', 'd2o', 'h2o', 'water', 'heavy water', 'vacuum'];
  environments.forEach(env => {
    if (lowerMessage.includes(env)) {
      params.push({
        key: `environment_${env}_${timestamp.getTime()}`,
        label: 'Environment',
        value: env === 'h2o' ? 'H₂O' : env === 'd2o' ? 'D₂O' : env,
        timestamp,
        confidence: 'high',
        category: 'environment'
      });
    }
  });

  // Extract substrate
  const substrates = ['silicon', 'glass', 'quartz', 'sapphire'];
  substrates.forEach(sub => {
    if (lowerMessage.includes(sub + ' substrate') || lowerMessage.includes(sub + ' wafer')) {
      params.push({
        key: `substrate_${sub}_${timestamp.getTime()}`,
        label: 'Substrate',
        value: sub,
        timestamp,
        confidence: 'high',
        category: 'substrate'
      });
    }
  });

  return params;
}

// IMPROVED: Deduplicate parameters by value + category
function deduplicateParams(params: CollectedParameter[]): CollectedParameter[] {
  const seen = new Map<string, CollectedParameter>();
  
  // Iterate in order so we keep the LATEST occurrence of each unique param
  for (const param of params) {
    const dedupeKey = `${param.category}::${param.value.toLowerCase().trim()}`;
    // Always overwrite with the newer entry (later in the array = more recent)
    seen.set(dedupeKey, param);
  }
  
  return Array.from(seen.values());
}

// NEW: Smart suggestions generator
function generateSmartSuggestions(lastMessage: string, role: 'user' | 'assistant'): SmartSuggestion[] {
  const lowerMessage = lastMessage.toLowerCase();

  // If AI is asking about materials
  if (role === 'assistant' && (lowerMessage.includes('material') || lowerMessage.includes('layer'))) {
    return [
      { text: 'PMMA polymer (SLD: 1.0)', category: 'material', confidence: 0.9 },
      { text: 'Silicon dioxide (SLD: 3.47)', category: 'material', confidence: 0.9 },
      { text: 'Gold (SLD: 4.5)', category: 'material', confidence: 0.8 },
      { text: 'Polystyrene (SLD: 1.04)', category: 'material', confidence: 0.8 }
    ];
  }

  // If AI is asking about thickness
  if (role === 'assistant' && (lowerMessage.includes('thick') || lowerMessage.includes('dimension'))) {
    return [
      { text: '50 Å (thin layer)', category: 'thickness', confidence: 0.9 },
      { text: '100 Å (medium)', category: 'thickness', confidence: 0.9 },
      { text: '200 Å (thick)', category: 'thickness', confidence: 0.8 },
      { text: '500 Å (very thick)', category: 'thickness', confidence: 0.7 }
    ];
  }

  // If AI is asking about environment
  if (role === 'assistant' && (lowerMessage.includes('environment') || lowerMessage.includes('solvent'))) {
    return [
      { text: 'Air', category: 'environment', confidence: 0.9 },
      { text: 'D₂O (heavy water)', category: 'environment', confidence: 0.9 },
      { text: 'H₂O (water)', category: 'environment', confidence: 0.8 },
      { text: 'Vacuum', category: 'environment', confidence: 0.7 }
    ];
  }

  // If AI is asking about analysis type
  if (role === 'assistant' && (lowerMessage.includes('analysis') || lowerMessage.includes('measurement'))) {
    return [
      { text: 'Quick analysis (5 min)', category: 'analysis', confidence: 0.9 },
      { text: 'Detailed fitting (20 min)', category: 'analysis', confidence: 0.8 },
      { text: 'High precision (1 hour)', category: 'analysis', confidence: 0.7 },
      { text: 'Parameter optimization', category: 'analysis', confidence: 0.6 }
    ];
  }

  // Default suggestions for general conversation
  if (role === 'assistant') {
    return [
      { text: '3-layer polymer film', category: 'quick', confidence: 0.7 },
      { text: 'Silicon substrate', category: 'quick', confidence: 0.7 },
      { text: 'Quick test analysis', category: 'analysis', confidence: 0.8 },
      { text: 'Need help with setup', category: 'quick', confidence: 0.6 }
    ];
  }

  return [];
}

// NEW: Parameter Sidebar Component
function ParameterSidebar({ 
  parameters, 
  isOpen, 
  onToggle 
}: { 
  parameters: CollectedParameter[]; 
  isOpen: boolean; 
  onToggle: () => void; 
}) {
  const groupedParams = parameters.reduce((groups, param) => {
    if (!groups[param.category]) groups[param.category] = [];
    groups[param.category].push(param);
    return groups;
  }, {} as Record<string, CollectedParameter[]>);

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'material': return '🧪';
      case 'thickness': return '📏';
      case 'environment': return '🌡️';
      case 'substrate': return '🔸';
      case 'analysis': return '⚙️';
      default: return '📋';
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'material': return '#10b981';
      case 'thickness': return '#3b82f6';
      case 'environment': return '#f59e0b';
      case 'substrate': return '#8b5cf6';
      case 'analysis': return '#ef4444';
      default: return '#6b7280';
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: '56px',
      left: 0,
      bottom: 0,
      width: '280px',
      background: '#0d0d0d',
      borderRight: '1px solid #2a2a2a',
      transform: isOpen ? 'translateX(0)' : 'translateX(-100%)',
      transition: 'transform 0.3s ease',
      zIndex: 30,
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Header */}
      <div style={{ 
        padding: '16px', 
        borderBottom: '1px solid #2a2a2a',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <div style={{ fontSize: '12px', fontWeight: 600, color: 'white' }}>Collected Parameters</div>
          <div style={{ fontSize: '10px', color: '#666', marginTop: '2px' }}>
            {parameters.length} unique parameters detected
          </div>
        </div>
        <button 
          onClick={onToggle}
          style={{ 
            background: 'none', 
            border: '1px solid #333', 
            color: '#888', 
            cursor: 'pointer', 
            fontSize: '10px',
            padding: '4px 8px',
            fontFamily: 'monospace'
          }}>
          ←
        </button>
      </div>

      {/* Parameters List */}
      <div style={{ flex: 1, overflow: 'auto', padding: '12px' }}>
        {Object.keys(groupedParams).length === 0 ? (
          <div style={{ 
            color: '#666', 
            fontSize: '11px', 
            textAlign: 'center', 
            padding: '40px 20px' 
          }}>
            Start chatting to collect parameters automatically
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {Object.entries(groupedParams).map(([category, params]) => (
              <div key={category}>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '6px', 
                  marginBottom: '8px' 
                }}>
                  <span style={{ fontSize: '12px' }}>{getCategoryIcon(category)}</span>
                  <span style={{ 
                    fontSize: '11px', 
                    fontWeight: 600, 
                    color: getCategoryColor(category),
                    textTransform: 'capitalize'
                  }}>
                    {category}
                  </span>
                  <span style={{ 
                    fontSize: '9px', 
                    color: '#666', 
                    background: '#1a1a1a',
                    padding: '2px 6px',
                    borderRadius: '8px'
                  }}>
                    {params.length}
                  </span>
                </div>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  {params.slice(-3).map((param) => (
                    <div key={param.key} style={{ 
                      padding: '8px 12px', 
                      background: '#1a1a1a', 
                      border: '1px solid #333',
                      borderRadius: '4px'
                    }}>
                      <div style={{ 
                        fontSize: '11px', 
                        color: '#ccc',
                        fontFamily: 'monospace'
                      }}>
                        {param.value}
                      </div>
                      <div style={{ 
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginTop: '4px'
                      }}>
                        <span style={{ 
                          fontSize: '9px', 
                          color: '#666' 
                        }}>
                          {param.timestamp.toLocaleTimeString()}
                        </span>
                        <div style={{ 
                          width: '6px', 
                          height: '6px', 
                          borderRadius: '50%',
                          background: 
                            param.confidence === 'high' ? '#10b981' :
                            param.confidence === 'medium' ? '#f59e0b' : '#ef4444'
                        }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Summary */}
      {parameters.length > 0 && (
        <div style={{ 
          padding: '12px', 
          borderTop: '1px solid #2a2a2a',
          background: '#0a0a0a'
        }}>
          <div style={{ fontSize: '10px', color: '#666', marginBottom: '8px' }}>
            READY TO ANALYZE
          </div>
          <div style={{ fontSize: '9px', color: '#888', lineHeight: 1.4 }}>
            {Object.keys(groupedParams).map(cat => 
              `${cat}: ${groupedParams[cat].length}`
            ).join(' • ')}
          </div>
        </div>
      )}
    </div>
  );
}

// IMPROVED: Smart Suggestions Component - accepts disabled prop to hide stale suggestions
function SmartSuggestions({ 
  suggestions, 
  onSuggestionClick,
  isLatest
}: { 
  suggestions: SmartSuggestion[]; 
  onSuggestionClick: (text: string) => void;
  isLatest: boolean;
}) {
  if (suggestions.length === 0) return null;

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'material': return '#10b981';
      case 'thickness': return '#3b82f6';
      case 'environment': return '#f59e0b';
      case 'analysis': return '#ef4444';
      case 'quick': return '#8b5cf6';
      default: return '#6b7280';
    }
  };

  // For non-latest messages, show faded, non-interactive pills
  if (!isLatest) {
    return (
      <div style={{ marginTop: '12px', opacity: 0.35, pointerEvents: 'none' }}>
        <div style={{ 
          fontSize: '10px', 
          color: '#666', 
          marginBottom: '8px',
          textTransform: 'uppercase',
          letterSpacing: '0.05em'
        }}>
          💡 Suggestions (past)
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
          {suggestions.slice(0, 6).map((suggestion, i) => (
            <div
              key={i}
              style={{
                padding: '8px 12px',
                background: '#1a1a1a',
                border: '1px solid #2a2a2a',
                borderRadius: '16px',
                color: '#888',
                fontSize: '11px',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <div style={{ 
                width: '6px', height: '6px', borderRadius: '50%',
                background: getCategoryColor(suggestion.category),
                opacity: 0.4
              }} />
              {suggestion.text}
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div style={{ marginTop: '12px' }}>
      <div style={{ 
        fontSize: '10px', 
        color: '#666', 
        marginBottom: '8px',
        textTransform: 'uppercase',
        letterSpacing: '0.05em'
      }}>
        💡 Smart Suggestions
      </div>
      <div style={{ 
        display: 'flex', 
        flexWrap: 'wrap', 
        gap: '8px' 
      }}>
        {suggestions.slice(0, 6).map((suggestion, i) => (
          <button
            key={i}
            onClick={() => onSuggestionClick(suggestion.text)}
            style={{
              padding: '8px 12px',
              background: '#1a1a1a',
              border: '1px solid #333',
              borderRadius: '16px',
              color: '#ccc',
              fontSize: '11px',
              cursor: 'pointer',
              transition: 'all 0.2s',
              fontFamily: 'inherit',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
            onMouseOver={e => {
              e.currentTarget.style.borderColor = getCategoryColor(suggestion.category);
              e.currentTarget.style.color = '#fff';
              e.currentTarget.style.background = '#252525';
            }}
            onMouseOut={e => {
              e.currentTarget.style.borderColor = '#333';
              e.currentTarget.style.color = '#ccc';
              e.currentTarget.style.background = '#1a1a1a';
            }}
          >
            <div style={{ 
              width: '6px', 
              height: '6px', 
              borderRadius: '50%',
              background: getCategoryColor(suggestion.category),
              opacity: suggestion.confidence
            }} />
            {suggestion.text}
            <span style={{ fontSize: '10px', opacity: 0.5 }}>↵</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// All your existing functions remain the same
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

async function callGenerateAPI(config: GenerationConfig, onProgress?: (message: string) => void): Promise<GenerateResponse> {
  const layers = buildLayersPayload(config);
  
  try {
    if (onProgress) onProgress('🚀 Starting neutron reflectivity analysis...');
    if (onProgress) onProgress('📊 Generating synthetic data and training model...');
    
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
    
    if (onProgress) onProgress('✅ Analysis complete! Processing results...');
    return response.json();
    
  } catch (error) {
    if (onProgress) onProgress(`❌ Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    throw error;
  }
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

// All your existing components remain exactly the same
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

      <button onClick={onToggle}
        style={{ position: 'absolute', bottom: '-32px', left: '50%', transform: 'translateX(-50%)', background: '#1a1a1a', border: '1px solid #333', borderTop: 'none', color: '#888', padding: '4px 16px', cursor: 'pointer', fontSize: '10px', fontFamily: 'monospace', display: 'flex', alignItems: 'center', gap: '6px' }}>
        <span style={{ transform: isCollapsed ? 'rotate(180deg)' : 'rotate(0)', transition: 'transform 0.3s' }}>▲</span>
        {isCollapsed ? 'SHOW PARAMS' : 'HIDE PARAMS'}
      </button>
    </div>
  );
}

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

export default function ChatPage() {
  // All your existing state variables
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
  
  // NEW: Enhanced feature state variables
  const [collectedParams, setCollectedParams] = useState<CollectedParameter[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  useEffect(() => {
    const testConnection = async () => {
      try {
        const health = await pyreflectAPI.healthCheck();
        console.log('✅ PyReflect API connected:', health);
      } catch (error) {
        console.log('❌ API connection failed:', error);
      }
    };
    
    testConnection();
  }, []);

  // IMPROVED: Compute deduplicated params for the sidebar
  const deduplicatedParams = deduplicateParams(collectedParams);

  // IMPROVED: Find the index of the last assistant message for suggestion rendering
  const lastAssistantIndex = (() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'assistant' && messages[i].suggestions && messages[i].suggestions!.length > 0) {
        return i;
      }
    }
    return -1;
  })();

  // All your existing callback functions remain the same
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
      const result = await callGenerateAPI(config, (progressMessage) => {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: progressMessage
        }]);
      });
      
      const duration = Date.now() - start;
      setLastDuration(duration);
      setGraphData(result);
      
      const historyItem: HistoryItem = {
        id: result.model_id ?? `generation_${Date.now()}`,
        config,
        result,
        timestamp: new Date(),
        duration
      };
      setHistory(prev => [historyItem, ...prev]);
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `🎉 **Generation complete!** (${formatDuration(duration)})\n\nModel ID: \`${result.model_id}\`\nR²: ${result.metrics.r2.toFixed(4)} • MSE: ${result.metrics.mse.toFixed(4)}`
      }]);
      
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Unknown error';
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `💥 **Generation failed:** ${msg}\n\nYou can try again or adjust your parameters.` 
      }]);
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

  // IMPROVED: sendMessage now auto-sends suggestion clicks and deduplicates params
  const sendMessage = useCallback(async (messageText?: string) => {
    const text = messageText || input;
    if (!text.trim() || isLoading) return;
    if (isTestCommand(text)) { setInput(''); handleQuickTest(); return; }

    const userMessage: MessageType = { role: 'user', content: text };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Extract parameters from user message (dedup happens at render via deduplicateParams)
    const newParams = extractParametersFromMessage(text, 'user');
    setCollectedParams(prev => [...prev, ...newParams]);

    try {
      const apiKey = process.env.NEXT_PUBLIC_OPENROUTER_API_KEY;
      if (!apiKey) throw new Error('API key not found');

      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'z-ai/glm-4.5-air:free',
          reasoning: {
            enabled: true
          },
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
      
      // Generate smart suggestions for assistant response
      const suggestions = generateSmartSuggestions(assistantContent, 'assistant');
      
      const assistantMessage: MessageType = { 
        role: 'assistant', 
        content: assistantContent,
        suggestions 
      };
      setMessages(prev => [...prev, assistantMessage]);

      // Extract parameters from assistant message
      const assistantParams = extractParametersFromMessage(assistantContent, 'assistant');
      setCollectedParams(prev => [...prev, ...assistantParams]);

      const config = extractJSON(assistantContent);
      if (config?.ready_to_generate) {
        const genConfig: GenerationConfig = { 
          substrate: config.substrate || 'silicon', 
          layers: config.layers || [], 
          environment: config.environment || 'air', 
          numCurves: 100, 
          epochs: 10 
        };
        setPendingConfig(genConfig);
        await handleGeneration(genConfig);
      }
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `Error: ${error instanceof Error ? error.message : 'Unknown'}` 
      }]);
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, messages, handleQuickTest, handleGeneration]);

  const handleReset = () => {
    setMessages([]);
    setGraphData(null);
    setPendingConfig(null);
    setParamsCollapsed(false);
    setUploadedFiles([]);
    setCollectedParams([]);
  };

  const handleHistorySelect = (item: HistoryItem) => {
    setGraphData(item.result);
    setPendingConfig(item.config);
    setHistoryOpen(false);
  };

  // IMPROVED: Suggestion clicks now auto-send instead of just populating input
  const handleSuggestionClick = useCallback((suggestionText: string) => {
    sendMessage(suggestionText);
  }, [sendMessage]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', backgroundColor: '#0a0a0a', fontFamily: "'JetBrains Mono', 'SF Mono', monospace" }}>
      {/* NEW: Parameter Sidebar — now uses deduplicated params */}
      <ParameterSidebar 
        parameters={deduplicatedParams}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />

      {/* Header with sidebar toggle */}
      <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 20px', height: '56px', borderBottom: '1px solid #2a2a2a' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            style={{
              width: '32px', 
              height: '32px', 
              background: sidebarOpen ? '#10b981' : '#333', 
              border: 'none',
              color: sidebarOpen ? 'black' : '#888',
              cursor: 'pointer',
              fontSize: '16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: '4px'
            }}
          >
            📋
          </button>
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
          <button
            onClick={async () => {
              try {
                const health = await pyreflectAPI.healthCheck();
                alert('✅ API Connected!\n\n' + JSON.stringify(health, null, 2));
              } catch (error) {
                alert('❌ API Connection Failed!\n\n' + (error instanceof Error ? error.message : String(error)));
              }
            }}
            style={{ color: '#888', background: 'none', border: '1px solid #333', padding: '6px 12px', cursor: 'pointer', fontSize: '11px', fontFamily: 'inherit', textTransform: 'uppercase' }}>
            🔗 Test API
          </button>
          <button onClick={handleReset}
            style={{ color: '#888', background: 'none', border: '1px solid #333', padding: '6px 12px', cursor: 'pointer', fontSize: '11px', fontFamily: 'inherit', textTransform: 'uppercase' }}>
            New Chat
          </button>
        </div>
      </header>

      {/* Main Content with adjusted layout */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Left: Chat - adjusted width when sidebar is open */}
        <div style={{ 
          width: sidebarOpen ? 'calc(45% - 140px)' : '45%',
          marginLeft: sidebarOpen ? '280px' : '0',
          transition: 'all 0.3s ease',
          display: 'flex', 
          flexDirection: 'column', 
          borderRight: '1px solid #2a2a2a' 
        }}>
          <div style={{ flex: 1, overflow: 'auto', padding: '16px' }}>
            {messages.length === 0 ? <WelcomeScreen onSuggestionClick={sendMessage} /> : (
              <div>
                {messages.map((m, i) => (
                  <div key={i}>
                    <Message role={m.role} content={m.content} />
                    {/* IMPROVED: Only show interactive suggestions on the LATEST assistant message */}
                    {m.role === 'assistant' && m.suggestions && m.suggestions.length > 0 && (
                      <SmartSuggestions 
                        suggestions={m.suggestions}
                        onSuggestionClick={handleSuggestionClick}
                        isLatest={i === lastAssistantIndex}
                      />
                    )}
                  </div>
                ))}
                {isLoading && <div style={{ padding: '16px', color: '#666', fontSize: '12px' }}>Thinking...</div>}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Chat Input - same as before */}
          <div style={{ borderTop: '1px solid #2a2a2a', padding: '12px 16px' }}>
            {uploadedFiles.length > 0 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '12px' }}>
                {uploadedFiles.map(file => (
                  <div key={file.id}
                    style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 10px', background: '#1a1a1a', border: '1px solid #333', fontSize: '11px' }}>
                    <span style={{ color: file.preview ? '#10b981' : '#888' }}>
                      {file.name.endsWith('.csv') ? '📊' : file.name.endsWith('.txt') ? '📄' : '📄'}
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
                →
              </button>
            </div>

            <div style={{ fontSize: '10px', color: '#444', marginTop: '8px', textAlign: 'center' }}>
              Upload .csv, .txt, .dat, .json data files • AI will help analyze your reflectivity data
            </div>
          </div>
        </div>

        {/* Right: Results - same as before */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative', overflow: 'hidden' }}>
          <ParameterPanel config={pendingConfig} onChange={setPendingConfig} onGenerate={() => pendingConfig && handleGeneration(pendingConfig)} isGenerating={isGenerating} isCollapsed={paramsCollapsed} onToggle={() => setParamsCollapsed(!paramsCollapsed)} />

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

          <StatusBar history={history} isGenerating={isGenerating} lastDuration={lastDuration} onHistoryClick={() => setHistoryOpen(true)} />
        </div>
      </div>

      {/* All your existing modals remain the same */}
      <HistoryPanel history={history} isOpen={historyOpen} onClose={() => setHistoryOpen(false)} onSelect={handleHistorySelect} />

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