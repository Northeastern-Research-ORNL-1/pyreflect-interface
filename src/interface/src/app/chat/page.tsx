'use client';
import { pyreflectAPI } from '../../services/pyreflectAPI';
import { useState, useRef, useEffect, useCallback } from 'react';
import WelcomeScreen from './components/WelcomeScreen';
import Message from './components/Message';
import GraphDisplay from '@/components/GraphDisplay';
import { GenerateResponse } from '@/types';

// ============================================================
// INTERFACES
// ============================================================

interface MessageType {
  role: 'user' | 'assistant';
  content: string;
  suggestions?: SmartSuggestion[];
  isLatest?: boolean;
  model?: string;
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

// ============================================================
// AI ROUTER
// ============================================================

const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';

const MODEL_CHAIN = [
  'z-ai/glm-4.5-air:free',
  'meta-llama/llama-3.3-70b-instruct:free',
  'qwen/qwen3-coder:free',
  'openrouter/free',
] as const;

const SYSTEM_PROMPT = `You are PyReflect AI, a neutron reflectivity experiment assistant.

RULES:
1. When the user describes a sample, extract layer parameters and respond with ONLY a JSON code block.
2. Do NOT ask clarifying questions unless critical info is missing (substrate, layer material).
3. If the user says "generate", "run", "go", or "fit", output the JSON immediately.
4. Keep all text responses under 3 sentences.

OUTPUT FORMAT (follow exactly):
\`\`\`json
{
  "ready_to_generate": true,
  "substrate": "silicon",
  "environment": "air",
  "layers": [
    {"name": "layer_name", "sld": 1.0, "thickness": 100, "roughness": 3}
  ]
}
\`\`\`

SLD VALUES (use these, do not guess):
silicon: 2.07, sio2: 3.47, air: 0, d2o: 6.36, h2o: -0.56, gold: 4.5, pmma: 1.0, polystyrene: 1.04, titanium: -1.95

If user uploaded experimental data, add "fitToData": true
NEVER output anything outside the JSON block when generating.`;

function backoffMs(attempt: number): number {
  return Math.min(1000 * 2 ** attempt, 8000) + Math.random() * 500;
}

function compressHistory(
  msgs: Array<{ role: string; content: string }>,
  maxMessages: number = 6
): Array<{ role: string; content: string }> {
  if (msgs.length <= maxMessages) return msgs;
  const recent = msgs.slice(-maxMessages);
  const olderCount = msgs.length - maxMessages;
  return [
    { role: 'system', content: '[' + olderCount + ' earlier messages omitted. User is configuring a neutron reflectivity experiment.]' },
    ...recent,
  ];
}

async function diagnoseAPIKey(apiKey: string): Promise<{ ok: boolean; error?: string; status?: number; rateInfo?: string; modelResults?: string[] }> {
  const modelResults: string[] = [];
  let lastRateInfo = '';
  for (const model of MODEL_CHAIN) {
    const shortName = model.split('/').pop()?.replace(':free', '') || model;
    try {
      const res = await fetch(OPENROUTER_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'Bearer ' + apiKey,
          'HTTP-Referer': typeof window !== 'undefined' ? window.location.origin : 'http://localhost:3000',
          'X-Title': 'PyReflect AI',
        },
        body: JSON.stringify({ model, messages: [{ role: 'user', content: 'Say OK' }], max_tokens: 2, stream: false }),
      });
      const limit = res.headers.get('x-ratelimit-limit');
      const remaining = res.headers.get('x-ratelimit-remaining');
      if (limit || remaining) lastRateInfo = 'Limit: ' + (limit || '?') + ' | Remaining: ' + (remaining || '?');
      if (res.ok) {
        modelResults.push('✅ ' + shortName);
        return { ok: true, rateInfo: lastRateInfo, modelResults };
      } else {
        const body = await res.json().catch(() => ({}));
        modelResults.push('❌ ' + shortName + ': ' + (body?.error?.message || 'HTTP ' + res.status));
      }
    } catch (err: any) {
      modelResults.push('❌ ' + shortName + ': ' + (err.message || 'Network error'));
    }
  }
  return { ok: false, status: 429, error: 'All models returned errors', rateInfo: lastRateInfo, modelResults };
}

async function streamCompletion(
  model: string,
  messages: Array<{ role: string; content: string }>,
  apiKey: string,
  timeoutMs: number,
  onToken: (token: string) => void,
): Promise<string> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(OPENROUTER_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: 'Bearer ' + apiKey,
        'HTTP-Referer': typeof window !== 'undefined' ? window.location.origin : 'http://localhost:3000',
        'X-Title': 'PyReflect AI',
      },
      body: JSON.stringify({
        model, messages, temperature: 0.3, top_p: 0.85, max_tokens: 800, stream: true,
        ...(model.includes('glm') ? { reasoning: { enabled: true } } : {}),
      }),
      signal: controller.signal,
    });
    if (!res.ok) {
      let detail = 'HTTP ' + res.status;
      try { const err = await res.json(); detail = err?.error?.message || detail; } catch {}
      if (res.status === 401) throw new Error('AUTH_ERROR: ' + detail);
      if (res.status === 402) throw new Error('PAYMENT_ERROR: ' + detail);
      if (res.status === 429) {
        const isProvider = detail.toLowerCase().includes('provider');
        throw new Error(isProvider ? 'PROVIDER_LIMITED: ' + detail : 'RATE_LIMITED: ' + detail);
      }
      if (res.status === 503) throw new Error('SERVICE_UNAVAILABLE');
      throw new Error(detail);
    }
    const contentType = res.headers.get('content-type') || '';
    if (contentType.includes('application/json') && !contentType.includes('stream')) {
      const body = await res.json();
      if (body.error) throw new Error(body.error.message || 'Unknown API error');
      const content = body.choices?.[0]?.message?.content;
      if (content) { onToken(content); return content; }
      throw new Error('EMPTY_RESPONSE');
    }
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') break;
        try {
          const parsed = JSON.parse(data);
          if (parsed.error) throw new Error(parsed.error.message || 'Stream error');
          const token = parsed.choices?.[0]?.delta?.content;
          if (token) { fullText += token; onToken(token); }
        } catch (e: any) {
          if (e.message && !e.message.includes('JSON')) throw e;
        }
      }
    }
    if (!fullText.trim()) throw new Error('EMPTY_RESPONSE');
    return fullText;
  } finally {
    clearTimeout(timeout);
  }
}

async function sendToAI(
  messages: Array<{ role: string; content: string }>,
  apiKey: string,
  onToken: (token: string) => void,
  onModelSwitch?: (model: string, attempt: number) => void,
  onStatus?: (msg: string) => void,
): Promise<{ text: string; model: string }> {
  const TIMEOUT_MS = 90000;
  const MAX_RETRIES = 1;
  const errors: string[] = [];
  for (let m = 0; m < MODEL_CHAIN.length; m++) {
    const model = MODEL_CHAIN[m];
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        if (onModelSwitch) onModelSwitch(model, attempt);
        const shortName = model.split('/').pop()?.replace(':free', '') || model;
        console.log('[aiRouter] Trying ' + shortName + ' (attempt ' + (attempt + 1) + ')');
        const text = await streamCompletion(model, messages, apiKey, TIMEOUT_MS, onToken);
        console.log('[aiRouter] ✅ ' + shortName + ' succeeded');
        return { text, model };
      } catch (err: any) {
        const msg = err?.message || 'Unknown error';
        const shortName = model.split('/').pop()?.replace(':free', '') || model;
        console.warn('[aiRouter] ❌ ' + shortName + ': ' + msg);
        errors.push(shortName + ': ' + msg);
        if (msg.startsWith('AUTH_ERROR')) throw new Error('API key invalid or expired. Check .env.local.\n\n' + msg);
        if (msg.startsWith('PAYMENT_ERROR')) throw new Error('OpenRouter account issue (402).\n\n' + msg);
        if (msg.startsWith('PROVIDER_LIMITED')) {
          if (onStatus) onStatus(shortName + ' provider busy, switching...');
          break;
        }
        if (msg.startsWith('RATE_LIMITED')) {
          if (attempt === 0) {
            if (onStatus) onStatus('Rate limited — waiting 5s...');
            await new Promise(r => setTimeout(r, 5000));
            continue;
          }
          if (m < MODEL_CHAIN.length - 1) {
            if (onStatus) onStatus('Still limited — switching model...');
            await new Promise(r => setTimeout(r, 10000));
          }
          break;
        }
        if (err.name === 'AbortError') {
          if (onStatus) onStatus(shortName + ' timed out, trying next...');
          break;
        }
        if (attempt < MAX_RETRIES) {
          const delay = backoffMs(attempt);
          await new Promise(r => setTimeout(r, delay));
        }
      }
    }
  }
  throw new Error('All models failed:\n• ' + errors.join('\n• ') + '\n\nWait 60s and try again.');
}

// ============================================================
// CONSTANTS & HELPERS
// ============================================================

const SLD_VALUES: Record<string, number> = {
  'silicon': 2.07, 'si': 2.07, 'sio2': 3.47, 'silicon dioxide': 3.47, 'silica': 3.47,
  'air': 0, 'd2o': 6.36, 'heavy water': 6.36, 'h2o': -0.56, 'water': -0.56,
  'gold': 4.5, 'au': 4.5, 'titanium': -1.95, 'ti': -1.95,
  'pmma': 1.0, 'polystyrene': 1.04, 'ps': 1.04,
};

const TEST_CONFIGS: GenerationConfig[] = [
  { substrate: 'silicon', layers: [{ name: 'SiO2', thickness: 15, sld: 3.47, roughness: 3 }, { name: 'PMMA', thickness: 100, sld: 1.0, roughness: 5 }], environment: 'air', numCurves: 100, epochs: 10 },
  { substrate: 'silicon', layers: [{ name: 'Gold', thickness: 50, sld: 4.5, roughness: 2 }, { name: 'Polymer', thickness: 150, sld: 1.2, roughness: 8 }], environment: 'd2o', numCurves: 100, epochs: 10 }
];

function extractParametersFromMessage(message: string, role: 'user' | 'assistant'): CollectedParameter[] {
  const params: CollectedParameter[] = [];
  const timestamp = new Date();
  const lower = message.toLowerCase();
  const materialPatterns = [/(?:made of|material|layer.*?is|using)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)/gi, /([a-zA-Z]+)\s+(?:film|layer|coating)/gi, /(silicon|gold|pmma|polystyrene|titanium|sio2|polymer)\b/gi];
  materialPatterns.forEach(p => { [...message.matchAll(p)].forEach(m => { const mat = m[1]?.trim(); if (mat && mat.length > 2 && mat.length < 20) params.push({ key: 'material_' + mat + '_' + timestamp.getTime(), label: 'Material', value: mat, timestamp, confidence: SLD_VALUES[mat.toLowerCase()] ? 'high' : 'medium', category: 'material' }); }); });
  const thickPatterns = [/(\d+(?:\.\d+)?)\s*(?:nm|nanometers?|Å|angstroms?)\b/gi, /(?:thick(?:ness)?|layer).*?(\d+(?:\.\d+)?)/gi];
  thickPatterns.forEach(p => { [...message.matchAll(p)].forEach(m => { const t = m[1]; if (t && parseFloat(t) > 0 && parseFloat(t) < 10000) params.push({ key: 'thickness_' + t + '_' + timestamp.getTime(), label: 'Thickness', value: t + ' Å', timestamp, confidence: 'high', category: 'thickness' }); }); });
  ['air', 'd2o', 'h2o', 'water', 'heavy water', 'vacuum'].forEach(env => { if (lower.includes(env)) params.push({ key: 'environment_' + env + '_' + timestamp.getTime(), label: 'Environment', value: env === 'h2o' ? 'H₂O' : env === 'd2o' ? 'D₂O' : env, timestamp, confidence: 'high', category: 'environment' }); });
  ['silicon', 'glass', 'quartz', 'sapphire'].forEach(sub => { if (lower.includes(sub + ' substrate') || lower.includes(sub + ' wafer')) params.push({ key: 'substrate_' + sub + '_' + timestamp.getTime(), label: 'Substrate', value: sub, timestamp, confidence: 'high', category: 'substrate' }); });
  return params;
}

function deduplicateParams(params: CollectedParameter[]): CollectedParameter[] {
  const seen = new Map<string, CollectedParameter>();
  for (const p of params) seen.set(p.category + '::' + p.value.toLowerCase().trim(), p);
  return Array.from(seen.values());
}

function generateSmartSuggestions(lastMessage: string, role: 'user' | 'assistant'): SmartSuggestion[] {
  const l = lastMessage.toLowerCase();
  if (role === 'assistant' && (l.includes('material') || l.includes('layer'))) return [{ text: 'PMMA polymer (SLD: 1.0)', category: 'material', confidence: 0.9 }, { text: 'Silicon dioxide (SLD: 3.47)', category: 'material', confidence: 0.9 }, { text: 'Gold (SLD: 4.5)', category: 'material', confidence: 0.8 }, { text: 'Polystyrene (SLD: 1.04)', category: 'material', confidence: 0.8 }];
  if (role === 'assistant' && (l.includes('thick') || l.includes('dimension'))) return [{ text: '50 Å (thin layer)', category: 'thickness', confidence: 0.9 }, { text: '100 Å (medium)', category: 'thickness', confidence: 0.9 }, { text: '200 Å (thick)', category: 'thickness', confidence: 0.8 }, { text: '500 Å (very thick)', category: 'thickness', confidence: 0.7 }];
  if (role === 'assistant' && (l.includes('environment') || l.includes('solvent'))) return [{ text: 'Air', category: 'environment', confidence: 0.9 }, { text: 'D₂O (heavy water)', category: 'environment', confidence: 0.9 }, { text: 'H₂O (water)', category: 'environment', confidence: 0.8 }, { text: 'Vacuum', category: 'environment', confidence: 0.7 }];
  if (role === 'assistant') return [{ text: '3-layer polymer film', category: 'quick', confidence: 0.7 }, { text: 'Silicon substrate', category: 'quick', confidence: 0.7 }, { text: 'Quick test analysis', category: 'analysis', confidence: 0.8 }, { text: 'Need help with setup', category: 'quick', confidence: 0.6 }];
  return [];
}

// ============================================================
// COMPONENTS
// ============================================================

function ParameterSidebar({ parameters, isOpen, onToggle }: { parameters: CollectedParameter[]; isOpen: boolean; onToggle: () => void }) {
  const grouped = parameters.reduce((g, p) => { if (!g[p.category]) g[p.category] = []; g[p.category].push(p); return g; }, {} as Record<string, CollectedParameter[]>);
  const icon = (c: string) => { switch (c) { case 'material': return '🧪'; case 'thickness': return '📏'; case 'environment': return '🌡️'; case 'substrate': return '🔸'; case 'analysis': return '⚙️'; default: return '📋'; } };
  const color = (c: string) => { switch (c) { case 'material': return '#10b981'; case 'thickness': return '#3b82f6'; case 'environment': return '#f59e0b'; case 'substrate': return '#8b5cf6'; case 'analysis': return '#ef4444'; default: return '#6b7280'; } };
  return (
    <div style={{ position: 'fixed', top: '56px', left: 0, bottom: 0, width: '280px', background: '#0d0d0d', borderRight: '1px solid #2a2a2a', transform: isOpen ? 'translateX(0)' : 'translateX(-100%)', transition: 'transform 0.3s ease', zIndex: 30, display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '16px', borderBottom: '1px solid #2a2a2a', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div><div style={{ fontSize: '12px', fontWeight: 600, color: 'white' }}>Collected Parameters</div><div style={{ fontSize: '10px', color: '#666', marginTop: '2px' }}>{parameters.length} detected</div></div>
        <button onClick={onToggle} style={{ background: 'none', border: '1px solid #333', color: '#888', cursor: 'pointer', fontSize: '10px', padding: '4px 8px', fontFamily: 'monospace' }}>←</button>
      </div>
      <div style={{ flex: 1, overflow: 'auto', padding: '12px' }}>
        {Object.keys(grouped).length === 0 ? <div style={{ color: '#666', fontSize: '11px', textAlign: 'center', padding: '40px 20px' }}>Start chatting to collect parameters</div> : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {Object.entries(grouped).map(([cat, ps]) => (
              <div key={cat}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}><span style={{ fontSize: '12px' }}>{icon(cat)}</span><span style={{ fontSize: '11px', fontWeight: 600, color: color(cat), textTransform: 'capitalize' }}>{cat}</span><span style={{ fontSize: '9px', color: '#666', background: '#1a1a1a', padding: '2px 6px', borderRadius: '8px' }}>{ps.length}</span></div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  {ps.slice(-3).map(p => (<div key={p.key} style={{ padding: '8px 12px', background: '#1a1a1a', border: '1px solid #333', borderRadius: '4px' }}><div style={{ fontSize: '11px', color: '#ccc', fontFamily: 'monospace' }}>{p.value}</div><div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '4px' }}><span style={{ fontSize: '9px', color: '#666' }}>{p.timestamp.toLocaleTimeString()}</span><div style={{ width: '6px', height: '6px', borderRadius: '50%', background: p.confidence === 'high' ? '#10b981' : p.confidence === 'medium' ? '#f59e0b' : '#ef4444' }} /></div></div>))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function SmartSuggestions({ suggestions, onSuggestionClick, isLatest }: { suggestions: SmartSuggestion[]; onSuggestionClick: (t: string) => void; isLatest: boolean }) {
  if (suggestions.length === 0) return null;
  const col = (c: string) => { switch (c) { case 'material': return '#10b981'; case 'thickness': return '#3b82f6'; case 'environment': return '#f59e0b'; case 'analysis': return '#ef4444'; case 'quick': return '#8b5cf6'; default: return '#6b7280'; } };
  if (!isLatest) return (<div style={{ marginTop: '12px', opacity: 0.35, pointerEvents: 'none' }}><div style={{ fontSize: '10px', color: '#666', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>💡 Suggestions (past)</div><div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>{suggestions.slice(0, 6).map((s, i) => (<div key={i} style={{ padding: '8px 12px', background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: '16px', color: '#888', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '6px' }}><div style={{ width: '6px', height: '6px', borderRadius: '50%', background: col(s.category), opacity: 0.4 }} />{s.text}</div>))}</div></div>);
  return (<div style={{ marginTop: '12px' }}><div style={{ fontSize: '10px', color: '#666', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>💡 Smart Suggestions</div><div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>{suggestions.slice(0, 6).map((s, i) => (<button key={i} onClick={() => onSuggestionClick(s.text)} style={{ padding: '8px 12px', background: '#1a1a1a', border: '1px solid #333', borderRadius: '16px', color: '#ccc', fontSize: '11px', cursor: 'pointer', transition: 'all 0.2s', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: '6px' }} onMouseOver={e => { e.currentTarget.style.borderColor = col(s.category); e.currentTarget.style.color = '#fff'; }} onMouseOut={e => { e.currentTarget.style.borderColor = '#333'; e.currentTarget.style.color = '#ccc'; }}><div style={{ width: '6px', height: '6px', borderRadius: '50%', background: col(s.category), opacity: s.confidence }} />{s.text}<span style={{ fontSize: '10px', opacity: 0.5 }}>↵</span></button>))}</div></div>);
}

function getRandomTestConfig(): GenerationConfig { return TEST_CONFIGS[Math.floor(Math.random() * TEST_CONFIGS.length)]; }
function getSLD(m: string): number { return SLD_VALUES[m.toLowerCase().trim()] ?? 1.0; }

function buildLayersPayload(config: GenerationConfig): LayerConfig[] {
  const r: LayerConfig[] = [];
  r.push({ name: config.substrate.toLowerCase(), sld: getSLD(config.substrate), isld: 0, thickness: 0, roughness: 1.8 });
  if (config.substrate.toLowerCase().includes('silicon')) r.push({ name: 'siox', sld: 3.47, isld: 0, thickness: 12, roughness: 2 });
  config.layers.forEach((l, i) => { r.push({ name: l.name?.toLowerCase() || 'layer_' + (i + 1), sld: l.sld ?? getSLD(l.name || ''), isld: 0, thickness: l.thickness ?? 100, roughness: l.roughness ?? 5 }); });
  r.push({ name: config.environment.toLowerCase(), sld: getSLD(config.environment), isld: 0, thickness: 0, roughness: 0 });
  return r;
}

async function callGenerateAPI(config: GenerationConfig, onProgress?: (m: string) => void, hasRealData?: boolean): Promise<GenerateResponse> {
  const layers = buildLayersPayload(config);
  try {
    if (onProgress) onProgress(hasRealData ? '🔬 Fitting model to your experimental data...' : '🚀 Starting neutron reflectivity analysis...');
    if (onProgress) onProgress(hasRealData ? '📈 Optimizing layer parameters...' : '🧪 Generating synthetic curves...');
    const response = await fetch('http://127.0.0.1:8000/api/generate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ layers, generator: { numCurves: config.numCurves, numFilmLayers: layers.length - 2 }, training: { batchSize: 32, epochs: config.epochs, layers: 12, dropout: 0, latentDim: 16, aeEpochs: 50, mlpEpochs: 50 } })
    });
    if (!response.ok) { const err = await response.json(); throw new Error(err.detail || 'Generation failed'); }
    if (onProgress) onProgress('✅ Analysis complete!');
    return response.json();
  } catch (error) {
    if (onProgress) onProgress('❌ Failed: ' + (error instanceof Error ? error.message : 'Unknown'));
    throw error;
  }
}

function extractJSON(text: string): any | null {
  const fence = text.match(/```json?\s*([\s\S]*?)\s*```/);
  if (fence) { try { return normalizeConfig(JSON.parse(fence[1])); } catch { return null; } }
  const raw = text.match(/\{\s*"ready_to_generate"\s*:\s*true[\s\S]*\}/);
  if (raw) { try { return normalizeConfig(JSON.parse(raw[0])); } catch { return null; } }
  return null;
}

function normalizeConfig(c: any): any {
  if (!c?.ready_to_generate) return c;
  let sub = 'silicon';
  if (typeof c.substrate === 'string') sub = c.substrate;
  else if (c.substrate?.material) sub = c.substrate.material;
  else if (c.substrate?.name) sub = c.substrate.name;
  let env = 'air';
  if (typeof c.environment === 'string') env = c.environment;
  else if (c.environment?.material) env = c.environment.material.toLowerCase();
  else if (c.environment?.name) env = c.environment.name;
  const layers = (c.layers || []).map((l: any) => ({ name: l.name || l.material || 'layer', thickness: l.thickness || 100, sld: l.sld || 1, roughness: l.roughness || 5 }));
  return { ...c, substrate: sub, environment: env, layers };
}

function isTestCommand(t: string): boolean { const l = t.toLowerCase().trim(); return ['test', 'quick test', 'demo', 'sample', 'random'].some(p => l.includes(p)); }

function formatDuration(ms: number): string {
  const s = Math.floor(ms / 1000); const m = Math.floor(s / 60); const sec = s % 60; const t = Math.floor((ms % 1000) / 100);
  if (m > 0) return m + ':' + sec.toString().padStart(2, '0') + '.' + t;
  return sec + '.' + t + 's';
}

// ============================================================
// SAFE ERROR PARSER — never crashes on weird backend responses
// ============================================================
async function parseErrorResponse(response: Response): Promise<string> {
  try {
    const text = await response.text();
    try {
      const json = JSON.parse(text);
      if (typeof json.detail === 'string') return json.detail;
      if (json.detail) return JSON.stringify(json.detail);
      if (json.message) return json.message;
      return JSON.stringify(json);
    } catch {
      return text || ('HTTP ' + response.status);
    }
  } catch {
    return 'HTTP ' + response.status;
  }
}

// ============================================================
// UI COMPONENTS
// ============================================================

function LiveTimer({ startTime, isRunning }: { startTime: number | null; isRunning: boolean }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => { if (!isRunning || !startTime) return; const i = setInterval(() => setElapsed(Date.now() - startTime), 100); return () => clearInterval(i); }, [isRunning, startTime]);
  if (!isRunning) return null;
  return (<div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center', zIndex: 10 }}><div style={{ fontSize: '64px', fontWeight: 200, fontFamily: "'JetBrains Mono', monospace", color: '#10b981', textShadow: '0 0 40px rgba(16,185,129,0.3)' }}>{formatDuration(elapsed)}</div><div style={{ fontSize: '12px', color: '#666', textTransform: 'uppercase', letterSpacing: '0.1em', marginTop: '8px' }}>Generating curves...</div><div style={{ marginTop: '24px' }}><div style={{ width: '200px', height: '2px', background: '#1a1a1a', borderRadius: '1px', overflow: 'hidden' }}><div style={{ height: '100%', background: 'linear-gradient(90deg, #10b981, #059669)', animation: 'pulse 1.5s ease-in-out infinite', width: '40%' }} /></div></div><style>{`@keyframes pulse { 0%, 100% { transform: translateX(-100%); } 50% { transform: translateX(250%); } }`}</style></div>);
}

function ParameterPanel({ config, onChange, onGenerate, isGenerating, isCollapsed, onToggle }: { config: GenerationConfig | null; onChange: (c: GenerationConfig) => void; onGenerate: () => void; isGenerating: boolean; isCollapsed: boolean; onToggle: () => void }) {
  return (
    <div style={{ position: 'absolute', top: 0, left: 0, right: 0, background: '#0d0d0d', borderBottom: '1px solid #2a2a2a', transform: isCollapsed ? 'translateY(-100%)' : 'translateY(0)', transition: 'transform 0.3s ease', zIndex: 20, maxHeight: '45vh', overflow: 'auto' }}>
      <div style={{ padding: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}><span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.1em', color: '#888' }}>Parameters</span>{config && <span style={{ fontSize: '10px', color: '#10b981' }}>● Ready</span>}</div>
        {config ? (<>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '16px' }}>
            <div><label style={{ fontSize: '10px', color: '#666', textTransform: 'uppercase' }}>Substrate</label><input value={config.substrate} onChange={e => onChange({ ...config, substrate: e.target.value })} style={{ width: '100%', padding: '10px', background: '#1a1a1a', border: '1px solid #333', color: 'white', fontFamily: 'monospace', fontSize: '12px', marginTop: '4px' }} /></div>
            <div><label style={{ fontSize: '10px', color: '#666', textTransform: 'uppercase' }}>Environment</label><select value={config.environment} onChange={e => onChange({ ...config, environment: e.target.value })} style={{ width: '100%', padding: '10px', background: '#1a1a1a', border: '1px solid #333', color: 'white', fontFamily: 'monospace', fontSize: '12px', marginTop: '4px' }}><option value="air">Air</option><option value="d2o">D₂O</option><option value="h2o">H₂O</option></select></div>
          </div>
          <div style={{ marginBottom: '16px' }}><label style={{ fontSize: '10px', color: '#666', textTransform: 'uppercase' }}>Layers ({config.layers.length})</label><div style={{ marginTop: '8px', display: 'flex', flexDirection: 'column', gap: '8px' }}>{config.layers.map((layer, i) => (<div key={i} style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr', gap: '8px', padding: '10px', background: '#1a1a1a', border: '1px solid #252525' }}><input placeholder="Name" value={layer.name} onChange={e => { const l = [...config.layers]; l[i] = { ...layer, name: e.target.value }; onChange({ ...config, layers: l }); }} style={{ padding: '6px', background: '#0a0a0a', border: '1px solid #333', color: 'white', fontFamily: 'monospace', fontSize: '11px' }} /><input type="number" placeholder="Å" value={layer.thickness} onChange={e => { const l = [...config.layers]; l[i] = { ...layer, thickness: +e.target.value }; onChange({ ...config, layers: l }); }} style={{ padding: '6px', background: '#0a0a0a', border: '1px solid #333', color: 'white', fontFamily: 'monospace', fontSize: '11px' }} /><input type="number" step="0.1" placeholder="SLD" value={layer.sld} onChange={e => { const l = [...config.layers]; l[i] = { ...layer, sld: +e.target.value }; onChange({ ...config, layers: l }); }} style={{ padding: '6px', background: '#0a0a0a', border: '1px solid #333', color: 'white', fontFamily: 'monospace', fontSize: '11px' }} /><input type="number" placeholder="σ" value={layer.roughness} onChange={e => { const l = [...config.layers]; l[i] = { ...layer, roughness: +e.target.value }; onChange({ ...config, layers: l }); }} style={{ padding: '6px', background: '#0a0a0a', border: '1px solid #333', color: 'white', fontFamily: 'monospace', fontSize: '11px' }} /></div>))}</div></div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 2fr', gap: '12px' }}><div><label style={{ fontSize: '10px', color: '#666', textTransform: 'uppercase' }}>Curves</label><input type="number" value={config.numCurves} onChange={e => onChange({ ...config, numCurves: +e.target.value })} style={{ width: '100%', padding: '10px', background: '#1a1a1a', border: '1px solid #333', color: 'white', fontFamily: 'monospace', fontSize: '12px', marginTop: '4px' }} /></div><div><label style={{ fontSize: '10px', color: '#666', textTransform: 'uppercase' }}>Epochs</label><input type="number" value={config.epochs} onChange={e => onChange({ ...config, epochs: +e.target.value })} style={{ width: '100%', padding: '10px', background: '#1a1a1a', border: '1px solid #333', color: 'white', fontFamily: 'monospace', fontSize: '12px', marginTop: '4px' }} /></div><div style={{ display: 'flex', alignItems: 'flex-end' }}><button onClick={onGenerate} disabled={isGenerating} style={{ width: '100%', padding: '12px', background: isGenerating ? '#333' : '#10b981', color: isGenerating ? '#666' : 'black', border: 'none', fontFamily: 'monospace', fontSize: '11px', fontWeight: 600, textTransform: 'uppercase', cursor: isGenerating ? 'not-allowed' : 'pointer' }}>{isGenerating ? 'Generating...' : 'Generate'}</button></div></div>
        </>) : (<div style={{ color: '#666', fontSize: '12px', textAlign: 'center', padding: '20px' }}>Chat with AI or click Quick Test to configure</div>)}
      </div>
      <button onClick={onToggle} style={{ position: 'absolute', bottom: '-32px', left: '50%', transform: 'translateX(-50%)', background: '#1a1a1a', border: '1px solid #333', borderTop: 'none', color: '#888', padding: '4px 16px', cursor: 'pointer', fontSize: '10px', fontFamily: 'monospace', display: 'flex', alignItems: 'center', gap: '6px' }}><span style={{ transform: isCollapsed ? 'rotate(180deg)' : 'rotate(0)', transition: 'transform 0.3s' }}>▲</span>{isCollapsed ? 'SHOW PARAMS' : 'HIDE PARAMS'}</button>
    </div>
  );
}

function HistoryPanel({ history, isOpen, onClose, onSelect }: { history: HistoryItem[]; isOpen: boolean; onClose: () => void; onSelect: (i: HistoryItem) => void }) {
  return (<>{isOpen && <div onClick={onClose} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 40 }} />}<div style={{ position: 'fixed', top: 0, right: 0, bottom: 0, width: '320px', background: '#0d0d0d', borderLeft: '1px solid #2a2a2a', transform: isOpen ? 'translateX(0)' : 'translateX(100%)', transition: 'transform 0.3s ease', zIndex: 50, display: 'flex', flexDirection: 'column' }}><div style={{ padding: '16px', borderBottom: '1px solid #2a2a2a', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}><span style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Generation History</span><button onClick={onClose} style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer', fontSize: '18px' }}>×</button></div><div style={{ flex: 1, overflow: 'auto', padding: '12px' }}>{history.length === 0 ? (<div style={{ color: '#666', fontSize: '12px', textAlign: 'center', padding: '40px 20px' }}>No generations yet.</div>) : (<div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>{history.map((item, i) => (<button key={item.id} onClick={() => onSelect(item)} style={{ background: '#1a1a1a', border: '1px solid #333', padding: '12px', cursor: 'pointer', textAlign: 'left' }} onMouseOver={e => e.currentTarget.style.borderColor = '#10b981'} onMouseOut={e => e.currentTarget.style.borderColor = '#333'}><div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}><span style={{ fontSize: '11px', color: '#10b981', fontWeight: 600 }}>Run #{history.length - i}</span><span style={{ fontSize: '10px', color: '#666' }}>{formatDuration(item.duration)}</span></div><div style={{ fontSize: '11px', color: '#ccc', marginBottom: '4px' }}>{item.config.layers.map(l => l.name).join(' → ')}</div><div style={{ fontSize: '10px', color: '#666' }}>R² {item.result.metrics.r2.toFixed(3)} · MSE {item.result.metrics.mse.toFixed(4)}</div></button>))}</div>)}</div></div></>);
}

function StatusBar({ history, isGenerating, lastDuration, activeModel, onHistoryClick }: { history: HistoryItem[]; isGenerating: boolean; lastDuration: number | null; activeModel: string | null; onHistoryClick: () => void }) {
  const short = activeModel ? activeModel.split('/').pop()?.replace(':free', '') || activeModel : null;
  return (<div style={{ height: '40px', borderTop: '1px solid #2a2a2a', background: '#0d0d0d', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 16px', fontSize: '11px', fontFamily: "'JetBrains Mono', monospace" }}><div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}><span style={{ color: isGenerating ? '#10b981' : '#666' }}>{isGenerating ? '● RUNNING' : '○ IDLE'}</span>{lastDuration && !isGenerating && <span style={{ color: '#666' }}>Last: {formatDuration(lastDuration)}</span>}{short && <span style={{ color: '#444', fontSize: '10px' }}>via {short}</span>}</div><button onClick={onHistoryClick} style={{ background: 'none', border: '1px solid #333', color: '#888', padding: '4px 12px', cursor: 'pointer', fontSize: '10px', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: '6px' }}><span>📊</span> History ({history.length})</button></div>);
}

// ============================================================
// MAIN COMPONENT
// ============================================================

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
  const [hasUploadedData, setHasUploadedData] = useState(false);
  const [collectedParams, setCollectedParams] = useState<CollectedParameter[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeModel, setActiveModel] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);
  useEffect(() => { pyreflectAPI.healthCheck().then(h => console.log('✅ API:', h)).catch(e => console.log('❌ API:', e)); }, []);

  const deduplicatedParams = deduplicateParams(collectedParams);
  const lastAssistantIndex = (() => { for (let i = messages.length - 1; i >= 0; i--) { if (messages[i].role === 'assistant' && messages[i].suggestions?.length) return i; } return -1; })();

  // ============================================================
  // FILE UPLOAD — with safe error handling for all file types
  // ============================================================
  const handleFileUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;
    for (const file of Array.from(files)) {
      const reader = new FileReader();
      reader.onload = async (event) => {
        const newFile: UploadedFile = { id: Math.random().toString(36).substr(2, 9), name: file.name, size: file.size, type: file.type, data: event.target?.result || null };
        if (file.name.endsWith('.csv') || file.type === 'text/csv') {
          const text = event.target?.result as string;
          newFile.preview = text.split('\n').slice(0, 6).map(line => line.split(',').map(c => c.trim()));
        }
        setUploadedFiles(prev => [...prev, newFile]);

        // ---- NPY FILES → /api/upload ----
        if (file.name.toLowerCase().endsWith('.npy')) {
          try {
            const formData = new FormData();
            formData.append('files', file);
            // Infer role from filename
            let role = 'nr_train';
            const fname = file.name.toLowerCase();
            if (fname.includes('sld')) role = 'sld_train';

            else if (fname.includes('norm')) role = 'normalization_stats';
            formData.append('roles', role);
            const response = await fetch('http://localhost:8000/api/upload', { method: 'POST', body: formData, headers: { 'X-User-Id': 'anonymous' } });
            if (response.ok) {
              const result = await response.json();
              console.log('✅ NPY uploaded:', result);
              setHasUploadedData(true);
              setMessages(prev => [...prev, {
                role: 'assistant',
                content: '📊 **NPY file uploaded successfully!**\n\n- **File:** ' + file.name + '\n- **Role:** ' + role + '\n\nYour data is ready for analysis.',
                suggestions: [
                  { text: 'Fit this data', category: 'analysis', confidence: 0.9 },
                  { text: 'Set up a model for fitting', category: 'analysis', confidence: 0.8 },
                  { text: 'What substrate should I use?', category: 'quick', confidence: 0.7 }
                ]
              }]);
            } else {
              const errMsg = await parseErrorResponse(response);
              console.error('❌ NPY upload failed:', errMsg);
              setMessages(prev => [...prev, { role: 'assistant', content: '❌ **Upload failed:** ' + errMsg }]);
            }
          } catch (error) {
            console.error('❌ NPY upload error:', error);
            setMessages(prev => [...prev, { role: 'assistant', content: '❌ **Upload error:** ' + (error instanceof Error ? error.message : 'Network error') }]);
          }
        }

        // ---- CSV/TXT/DAT FILES → /api/upload (same endpoint, role: experimental_nr) ----
        else if (file.name.toLowerCase().endsWith('.csv') || file.name.toLowerCase().endsWith('.txt') || file.name.toLowerCase().endsWith('.dat')) {
          try {
            const formData = new FormData();
            formData.append('files', file);
            formData.append('roles', 'experimental_nr');
            const response = await fetch('http://localhost:8000/api/upload', { method: 'POST', body: formData, headers: { 'X-User-Id': 'anonymous' } });
            if (response.ok) {
              const result = await response.json();
              console.log('✅ CSV uploaded:', result);
              setHasUploadedData(true);
              // /api/upload returns { saved: [...], metadata: [...] }
              const meta = result.metadata?.[0] || {};
              setMessages(prev => [...prev, {
                role: 'assistant',
                content: '📊 **File uploaded successfully!**\n\n- **File:** ' + (meta.filename || file.name) + '\n- **Role:** ' + (meta.role || 'experimental_nr') + '\n\nYour data is ready for analysis.',
                suggestions: [
                  { text: 'Fit this data', category: 'analysis', confidence: 0.9 },
                  { text: 'Show me the Q-R plot', category: 'analysis', confidence: 0.8 },
                  { text: 'Set up a model for fitting', category: 'analysis', confidence: 0.8 },
                  { text: 'What substrate should I use?', category: 'quick', confidence: 0.7 }
                ]
              }]);
            } else {
              const errMsg = await parseErrorResponse(response);
              console.error('❌ CSV upload failed:', errMsg);
              setMessages(prev => [...prev, { role: 'assistant', content: '❌ **Upload failed:** ' + errMsg }]);
            }
          } catch (error) {
            console.error('❌ CSV upload error:', error);
            setMessages(prev => [...prev, { role: 'assistant', content: '❌ **Upload error:** ' + (error instanceof Error ? error.message : 'Network error') }]);
          }
        }
      };
      if (file.name.endsWith('.csv') || file.type === 'text/csv' || file.type.startsWith('text/')) reader.readAsText(file);
      else reader.readAsArrayBuffer(file);
    }
    e.target.value = '';
  }, []);

  const removeFile = useCallback((id: string) => { setUploadedFiles(prev => prev.filter(f => f.id !== id)); if (showFilePreview?.id === id) setShowFilePreview(null); }, [showFilePreview]);
  const formatFileSize = (b: number) => { if (b < 1024) return b + ' B'; if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KB'; return (b / (1024 * 1024)).toFixed(1) + ' MB'; };

  const handleGeneration = useCallback(async (config: GenerationConfig) => {
    setIsGenerating(true); setParamsCollapsed(true); const start = Date.now(); setGenerationStart(start);
    try {
      const result = await callGenerateAPI(config, msg => setMessages(prev => [...prev, { role: 'assistant', content: msg }]), hasUploadedData);
      const duration = Date.now() - start; setLastDuration(duration); setGraphData(result);
      setHistory(prev => [{ id: result.model_id ?? 'gen_' + Date.now(), config, result, timestamp: new Date(), duration }, ...prev]);
      setMessages(prev => [...prev, { role: 'assistant', content: '🎉 **Generation complete!** (' + formatDuration(duration) + ')\n\nModel ID: `' + result.model_id + '`\nR²: ' + result.metrics.r2.toFixed(4) + ' • MSE: ' + result.metrics.mse.toFixed(4) }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'assistant', content: '💥 **Generation failed:** ' + (error instanceof Error ? error.message : 'Unknown error') }]);
    } finally { setIsGenerating(false); setGenerationStart(null); }
  }, [hasUploadedData]);

  const handleQuickTest = useCallback(() => {
    const tc = getRandomTestConfig(); setPendingConfig(tc);
    const ls = tc.layers.map(l => l.name + ' (' + l.thickness + 'Å)').join(', ');
    setMessages(prev => [...prev, { role: 'user', content: '🧪 Quick Test' }, { role: 'assistant', content: '**Test Config:**\n- **Substrate:** ' + tc.substrate + '\n- **Layers:** ' + ls + '\n- **Environment:** ' + tc.environment }]);
    handleGeneration(tc);
  }, [handleGeneration]);

  const sendMessage = useCallback(async (messageText?: string) => {
    const text = messageText || input;
    if (!text.trim() || isLoading) return;
    if (isTestCommand(text)) { setInput(''); handleQuickTest(); return; }
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setInput(''); setIsLoading(true);
    setCollectedParams(prev => [...prev, ...extractParametersFromMessage(text, 'user')]);
    setMessages(prev => [...prev, { role: 'assistant', content: '', suggestions: [] }]);
    try {
      const apiKey = process.env.NEXT_PUBLIC_OPENROUTER_API_KEY;
      if (!apiKey) throw new Error('API key not found');
      const history = messages.map(m => ({ role: m.role, content: m.content }));
      const apiMessages = [{ role: 'system', content: SYSTEM_PROMPT }, ...compressHistory(history), { role: 'user', content: text }];
      const { text: fullResponse, model: usedModel } = await sendToAI(apiMessages, apiKey,
        token => { setMessages(prev => { const u = [...prev]; const i = u.length - 1; if (i >= 0 && u[i].role === 'assistant') u[i] = { ...u[i], content: u[i].content + token }; return u; }); },
        (model, attempt) => { setActiveModel(model); },
        statusMsg => { setMessages(prev => { const u = [...prev]; const i = u.length - 1; if (i >= 0 && u[i].role === 'assistant') u[i] = { ...u[i], content: '⏳ ' + statusMsg }; return u; }); }
      );
      setActiveModel(usedModel);
      const suggestions = generateSmartSuggestions(fullResponse, 'assistant');
      setMessages(prev => { const u = [...prev]; const i = u.length - 1; if (i >= 0 && u[i].role === 'assistant') u[i] = { ...u[i], content: fullResponse, suggestions, model: usedModel }; return u; });
      setCollectedParams(prev => [...prev, ...extractParametersFromMessage(fullResponse, 'assistant')]);
      const config = extractJSON(fullResponse);
      if (config?.ready_to_generate) {
        const gc: GenerationConfig = { substrate: config.substrate || 'silicon', layers: config.layers || [], environment: config.environment || 'air', numCurves: 100, epochs: 10 };
        setPendingConfig(gc); await handleGeneration(gc);
      }
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Unknown error';
      setMessages(prev => { const u = [...prev]; const i = u.length - 1; if (i >= 0 && u[i].role === 'assistant') u[i] = { ...u[i], content: '⚠️ ' + msg }; return u; });
    } finally { setIsLoading(false); }
  }, [input, isLoading, messages, handleQuickTest, handleGeneration]);

  const handleReset = () => { setMessages([]); setGraphData(null); setPendingConfig(null); setParamsCollapsed(false); setUploadedFiles([]); setCollectedParams([]); setHasUploadedData(false); setActiveModel(null); };
  const handleSuggestionClick = useCallback((t: string) => { sendMessage(t); }, [sendMessage]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', backgroundColor: '#0a0a0a', fontFamily: "'JetBrains Mono', 'SF Mono', monospace" }}>
      <ParameterSidebar parameters={deduplicatedParams} isOpen={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />
      <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 20px', height: '56px', borderBottom: '1px solid #2a2a2a' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button onClick={() => setSidebarOpen(!sidebarOpen)} style={{ width: '32px', height: '32px', background: sidebarOpen ? '#10b981' : '#333', border: 'none', color: sidebarOpen ? 'black' : '#888', cursor: 'pointer', fontSize: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '4px' }}>📋</button>
          <div style={{ width: '32px', height: '32px', background: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="black" strokeWidth="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg></div>
          <span style={{ color: 'white', fontWeight: 600, fontSize: '14px' }}>PYREFLECT AI</span>
          {activeModel && <span style={{ fontSize: '10px', color: '#444', background: '#1a1a1a', padding: '2px 8px', borderRadius: '4px', border: '1px solid #252525' }}>{activeModel.split('/').pop()?.replace(':free', '')}</span>}
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button onClick={handleQuickTest} disabled={isGenerating || isLoading} style={{ color: 'black', background: '#10b981', border: 'none', padding: '6px 12px', cursor: isGenerating ? 'not-allowed' : 'pointer', fontSize: '11px', fontFamily: 'inherit', textTransform: 'uppercase', fontWeight: 600, opacity: isGenerating ? 0.5 : 1 }}>🧪 Quick Test</button>
          <button onClick={async () => {
            try {
              const r: string[] = [];
              try { const h = await pyreflectAPI.healthCheck(); r.push('✅ PyReflect Backend: Connected\n' + JSON.stringify(h, null, 2)); } catch (e) { r.push('❌ Backend: ' + (e instanceof Error ? e.message : String(e))); }
              const k = process.env.NEXT_PUBLIC_OPENROUTER_API_KEY;
              if (!k) { r.push('❌ No API key'); } else {
                r.push('🔑 Key: ' + k.slice(0, 12) + '...' + k.slice(-4));
                try { const d = await diagnoseAPIKey(k); r.push(d.ok ? '✅ OpenRouter: Working!' : '❌ OpenRouter: ' + (d.error || 'Error')); if (d.modelResults?.length) r.push('Models:\n' + d.modelResults.join('\n')); if (d.rateInfo) r.push('📊 ' + d.rateInfo); } catch (de) { r.push('❌ Diag failed: ' + (de instanceof Error ? de.message : String(de))); }
              }
              alert(r.join('\n\n'));
            } catch (oe) { alert('Error: ' + (oe instanceof Error ? oe.message : String(oe))); }
          }} style={{ color: '#888', background: 'none', border: '1px solid #333', padding: '6px 12px', cursor: 'pointer', fontSize: '11px', fontFamily: 'inherit', textTransform: 'uppercase' }}>🔗 Test API</button>
          <button onClick={handleReset} style={{ color: '#888', background: 'none', border: '1px solid #333', padding: '6px 12px', cursor: 'pointer', fontSize: '11px', fontFamily: 'inherit', textTransform: 'uppercase' }}>New Chat</button>
        </div>
      </header>
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <div style={{ width: sidebarOpen ? 'calc(45% - 140px)' : '45%', marginLeft: sidebarOpen ? '280px' : '0', transition: 'all 0.3s ease', display: 'flex', flexDirection: 'column', borderRight: '1px solid #2a2a2a' }}>
          <div style={{ flex: 1, overflow: 'auto', padding: '16px' }}>
            {messages.length === 0 ? <WelcomeScreen onSuggestionClick={sendMessage} /> : (
              <div>
                {messages.map((m, i) => (<div key={i}><Message role={m.role} content={m.content} />{m.role === 'assistant' && m.suggestions && m.suggestions.length > 0 && <SmartSuggestions suggestions={m.suggestions} onSuggestionClick={handleSuggestionClick} isLatest={i === lastAssistantIndex} />}</div>))}
                {isLoading && messages.length > 0 && messages[messages.length - 1].content === '' && (<div style={{ padding: '16px', color: '#10b981', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}><span style={{ animation: 'pulse 1s ease-in-out infinite' }}>●</span>Connecting to {activeModel?.split('/').pop()?.replace(':free', '') || 'AI'}...</div>)}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
          <div style={{ borderTop: '1px solid #2a2a2a', padding: '12px 16px' }}>
            {uploadedFiles.length > 0 && (<div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '12px' }}>{uploadedFiles.map(f => (<div key={f.id} style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 10px', background: '#1a1a1a', border: '1px solid #333', fontSize: '11px' }}><span style={{ color: f.preview ? '#10b981' : '#888' }}>{f.name.endsWith('.csv') ? '📊' : f.name.endsWith('.npy') ? '🔢' : '📄'}</span><button onClick={() => f.preview && setShowFilePreview(f)} style={{ background: 'none', border: 'none', color: '#ccc', cursor: f.preview ? 'pointer' : 'default', padding: 0, fontFamily: 'inherit', fontSize: '11px' }}>{f.name.length > 20 ? f.name.slice(0, 17) + '...' : f.name}</button><span style={{ color: '#666', fontSize: '10px' }}>{formatFileSize(f.size)}</span><button onClick={() => removeFile(f.id)} style={{ background: 'none', border: 'none', color: '#666', cursor: 'pointer', padding: '0 0 0 4px', fontSize: '14px', lineHeight: 1 }}>×</button></div>))}</div>)}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <input type="file" ref={fileInputRef} onChange={handleFileUpload} multiple accept=".csv,.txt,.dat,.json,.xml,.npy" style={{ display: 'none' }} />
              <button onClick={() => fileInputRef.current?.click()} style={{ width: '36px', height: '36px', background: 'none', border: '1px solid #333', color: '#888', cursor: 'pointer', fontSize: '18px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }} onMouseOver={e => { e.currentTarget.style.borderColor = '#10b981'; e.currentTarget.style.color = '#10b981'; }} onMouseOut={e => { e.currentTarget.style.borderColor = '#333'; e.currentTarget.style.color = '#888'; }}>+</button>
              <input type="text" value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendMessage()} placeholder={uploadedFiles.length > 0 ? 'Describe your data or ask a question...' : 'Ask anything'} disabled={isLoading || isGenerating} style={{ flex: 1, padding: '10px 14px', background: '#1a1a1a', border: '1px solid #333', color: 'white', fontFamily: 'inherit', fontSize: '13px', outline: 'none' }} />
              <button onClick={() => sendMessage()} disabled={isLoading || isGenerating || (!input.trim() && !uploadedFiles.length)} style={{ width: '36px', height: '36px', background: (input.trim() || uploadedFiles.length) ? '#10b981' : '#333', border: 'none', color: (input.trim() || uploadedFiles.length) ? 'black' : '#666', cursor: (input.trim() || uploadedFiles.length) ? 'pointer' : 'not-allowed', fontSize: '14px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>→</button>
            </div>
            <div style={{ fontSize: '10px', color: '#444', marginTop: '8px', textAlign: 'center' }}>Upload .csv, .txt, .dat, .npy, .json data files • AI will help analyze your reflectivity data</div>
          </div>
        </div>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative', overflow: 'hidden' }}>
          <ParameterPanel config={pendingConfig} onChange={setPendingConfig} onGenerate={() => pendingConfig && handleGeneration(pendingConfig)} isGenerating={isGenerating} isCollapsed={paramsCollapsed} onToggle={() => setParamsCollapsed(!paramsCollapsed)} />
          <div style={{ flex: 1, overflow: 'auto', padding: '16px', paddingTop: '48px', position: 'relative' }}>
            <LiveTimer startTime={generationStart} isRunning={isGenerating} />
            {graphData && !isGenerating ? <GraphDisplay data={graphData} /> : !isGenerating && (<div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#444', fontSize: '12px', textAlign: 'center' }}><div><div style={{ fontSize: '48px', marginBottom: '16px', opacity: 0.3 }}>◇</div><div>Click <strong>Quick Test</strong> or chat with AI</div></div></div>)}
          </div>
          <StatusBar history={history} isGenerating={isGenerating} lastDuration={lastDuration} activeModel={activeModel} onHistoryClick={() => setHistoryOpen(true)} />
        </div>
      </div>
      <HistoryPanel history={history} isOpen={historyOpen} onClose={() => setHistoryOpen(false)} onSelect={item => { setGraphData(item.result); setPendingConfig(item.config); setHistoryOpen(false); }} />
      {showFilePreview && (<><div onClick={() => setShowFilePreview(null)} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 60 }} /><div style={{ position: 'fixed', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', background: '#0d0d0d', border: '1px solid #333', padding: '20px', zIndex: 70, maxWidth: '80vw', maxHeight: '70vh', overflow: 'auto', minWidth: '400px' }}><div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}><div><div style={{ fontSize: '14px', fontWeight: 600, color: 'white' }}>{showFilePreview.name}</div><div style={{ fontSize: '11px', color: '#666', marginTop: '2px' }}>{formatFileSize(showFilePreview.size)} • CSV Preview</div></div><button onClick={() => setShowFilePreview(null)} style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer', fontSize: '20px' }}>×</button></div>{showFilePreview.preview && (<div style={{ overflow: 'auto' }}><table style={{ borderCollapse: 'collapse', fontSize: '11px', fontFamily: 'monospace' }}><thead><tr>{showFilePreview.preview[0]?.map((c, i) => (<th key={i} style={{ padding: '8px 12px', background: '#1a1a1a', border: '1px solid #333', color: '#10b981', textAlign: 'left', fontWeight: 600 }}>{c}</th>))}</tr></thead><tbody>{showFilePreview.preview.slice(1).map((row, i) => (<tr key={i}>{row.map((c, j) => (<td key={j} style={{ padding: '6px 12px', border: '1px solid #252525', color: '#ccc' }}>{c}</td>))}</tr>))}</tbody></table></div>)}<div style={{ marginTop: '16px', display: 'flex', gap: '8px' }}><button onClick={() => setShowFilePreview(null)} style={{ flex: 1, padding: '10px', background: '#1a1a1a', border: '1px solid #333', color: '#888', cursor: 'pointer', fontFamily: 'inherit', fontSize: '11px', textTransform: 'uppercase' }}>Close</button><button style={{ flex: 1, padding: '10px', background: '#10b981', border: 'none', color: 'black', cursor: 'pointer', fontFamily: 'inherit', fontSize: '11px', textTransform: 'uppercase', fontWeight: 600 }}>Use This Data</button></div></div></>)}
    </div>
  );
}
