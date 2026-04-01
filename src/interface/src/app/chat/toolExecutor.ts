import type { GenerateResponse } from '@/types';

export interface ToolCallbacks {
  onProgress: (msg: string) => void;
  onGraphData: (data: GenerateResponse) => void;
  setIsGenerating: (v: boolean) => void;
}

export interface ToolExecutionResult {
  content: string;           // JSON string returned to the LLM as the tool result
  graphData?: GenerateResponse;
}

// ---------------------------------------------------------------------------
// Shared SSE reader for /api/generate/stream
// ---------------------------------------------------------------------------

async function readGenerateStream(
  response: Response,
  onProgress?: (msg: string) => void,
): Promise<GenerateResponse> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');
  const decoder = new TextDecoder();
  let buffer = '';
  let result: GenerateResponse | null = null;
  let eventType = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        if (eventType === 'log' && onProgress) onProgress(data);
        else if (eventType === 'error')
          throw new Error(typeof data === 'string' ? data : JSON.stringify(data));
        else if (eventType === 'result') result = data as GenerateResponse;
      }
    }
  }
  if (!result) throw new Error('No result received from server');
  return result;
}

// ---------------------------------------------------------------------------
// run_inference
// ---------------------------------------------------------------------------

async function runInference(
  args: { dataSource: string; layers?: Array<{ name: string; sld: number; thickness: number; roughness: number }> },
  callbacks: ToolCallbacks,
): Promise<ToolExecutionResult> {
  callbacks.setIsGenerating(true);
  try {
    let payload: Record<string, unknown>;

    if (args.dataSource === 'real') {
      callbacks.onProgress('Running inference on uploaded experimental NR data...');
      payload = {
        layers: [
          { name: 'substrate', sld: 2.07, isld: 0, thickness: 0, roughness: 5 },
          { name: 'siox', sld: 3.47, isld: 0, thickness: 15, roughness: 3 },
          { name: 'film', sld: 1.0, isld: 0, thickness: 100, roughness: 10 },
          { name: 'air', sld: 0, isld: 0, thickness: 0, roughness: 0 },
        ],
        generator: { numCurves: 10, numFilmLayers: 1 },
        training: {
          batchSize: 64, epochs: 20, layers: 6, dropout: 0.0873,
          latentDim: 16, aeEpochs: 50, mlpEpochs: 50,
        },
        dataSource: 'real',
        mode: 'infer',
      };
    } else {
      callbacks.onProgress('Running synthetic inference...');
      const filmLayers = (args.layers || []).map((l) => ({
        name: l.name, sld: l.sld, isld: 0, thickness: l.thickness, roughness: l.roughness,
      }));
      const allLayers = [
        { name: 'substrate', sld: 2.07, isld: 0, thickness: 0, roughness: 5 },
        ...filmLayers,
        { name: 'air', sld: 0, isld: 0, thickness: 0, roughness: 0 },
      ];
      payload = {
        layers: allLayers,
        generator: { numCurves: 100, numFilmLayers: filmLayers.length },
        training: {
          batchSize: 32, epochs: 10, layers: 12, dropout: 0.0,
          latentDim: 16, aeEpochs: 50, mlpEpochs: 50,
        },
        mode: 'infer',
      };
    }

    const response = await fetch('http://127.0.0.1:8000/api/generate/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const errBody = await response.text();
      throw new Error('Generate API returned ' + response.status + ': ' + errBody);
    }

    const result = await readGenerateStream(response, callbacks.onProgress);
    callbacks.onGraphData(result);

    const summary = {
      r2: result.metrics.r2,
      mse: result.metrics.mse,
      mae: result.metrics.mae,
      dataSource: args.dataSource,
      sld_points: result.sld.predicted.length,
    };

    return {
      content: JSON.stringify(summary),
      graphData: result,
    };
  } finally {
    callbacks.setIsGenerating(false);
  }
}

// ---------------------------------------------------------------------------
// get_backend_status
// ---------------------------------------------------------------------------

async function getBackendStatus(
  callbacks: ToolCallbacks,
): Promise<ToolExecutionResult> {
  callbacks.onProgress('Checking backend status...');
  const response = await fetch('http://127.0.0.1:8000/api/status');
  if (!response.ok) throw new Error('Status API returned ' + response.status);
  const status = await response.json();

  const summary = {
    pyreflect_available: status.pyreflect_available,
    has_settings: status.has_settings,
    curve_files: status.curve_files || [],
    expt_files: status.expt_files || [],
    model_files: status.model_files || [],
    settings_status: status.settings_status,
  };

  return { content: JSON.stringify(summary) };
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

export async function executeToolCall(
  name: string,
  argsJson: string,
  callbacks: ToolCallbacks,
): Promise<ToolExecutionResult> {
  const args = argsJson ? JSON.parse(argsJson) : {};
  switch (name) {
    case 'run_inference':
      return runInference(args, callbacks);
    case 'get_backend_status':
      return getBackendStatus(callbacks);
    default:
      return { content: JSON.stringify({ error: 'Unknown tool: ' + name }) };
  }
}
