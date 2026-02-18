/**
 * pyreflectAPI.ts — 100% coverage test suite
 *
 * Strategy:
 *   • Global `fetch` is mocked via jest.fn().
 *   • Every public method + the private `request()` helper (tested
 *     indirectly through public methods) is exercised on success
 *     and failure paths.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

// ── Mocks ───────────────────────────────────────────────────────
const mockFetch = jest.fn();
(global as any).fetch = mockFetch;

// Suppress console.error / console.warn in test output
jest.spyOn(console, 'error').mockImplementation(() => {});
jest.spyOn(console, 'warn').mockImplementation(() => {});

// ── Imports (after mocking) ─────────────────────────────────────
import { pyreflectAPI } from './pyreflectAPI';

// ── Helpers ─────────────────────────────────────────────────────
/** Build a mock Response whose `.text()` returns the JSON-stringified body. */
function mockResponse(body: any, status = 200, ok = true): Response {
  const text = body === undefined || body === '' ? '' : JSON.stringify(body);
  return {
    ok,
    status,
    text: jest.fn().mockResolvedValue(text),
    json: jest.fn().mockResolvedValue(body),
    blob: jest.fn().mockResolvedValue(new Blob()),
    body: null,
    headers: new Headers(),
  } as unknown as Response;
}

/** Build a mock SSE ReadableStream from lines of text. */
function mockSSEResponse(lines: string[], status = 200): Response {
  const encoder = new TextEncoder();
  const chunks = lines.map((l) => encoder.encode(l + '\n'));
  let i = 0;

  const mockReader = {
    read: jest.fn().mockImplementation(async () => {
      if (i < chunks.length) {
        return { done: false, value: chunks[i++] };
      }
      return { done: true, value: undefined };
    }),
  };

  return {
    ok: status >= 200 && status < 300,
    status,
    body: { getReader: () => mockReader },
    headers: new Headers(),
  } as unknown as Response;
}

// ── Reset between tests ─────────────────────────────────────────
beforeEach(() => {
  mockFetch.mockReset();
});

// ================================================================
// request() — tested indirectly through healthCheck()
// ================================================================
describe('request() (via healthCheck)', () => {
  it('returns parsed JSON on success', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse({ status: 'ok' }));
    const res = await pyreflectAPI.healthCheck();
    expect(res).toEqual({ status: 'ok' });
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/health',
      expect.objectContaining({
        headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
      })
    );
  });

  it('throws on HTTP error with JSON message', async () => {
    mockFetch.mockResolvedValueOnce(
      mockResponse({ message: 'Unauthorized' }, 401, false)
    );
    await expect(pyreflectAPI.healthCheck()).rejects.toThrow('Unauthorized');
  });

  it('throws on HTTP error with JSON detail field', async () => {
    mockFetch.mockResolvedValueOnce(
      mockResponse({ detail: 'Forbidden' }, 403, false)
    );
    await expect(pyreflectAPI.healthCheck()).rejects.toThrow('Forbidden');
  });

  it('throws default message on HTTP error with non-JSON body', async () => {
    const resp = {
      ok: false,
      status: 500,
      text: jest.fn().mockResolvedValue('Internal Server Error'),
      headers: new Headers(),
    } as unknown as Response;
    mockFetch.mockResolvedValueOnce(resp);
    await expect(pyreflectAPI.healthCheck()).rejects.toThrow('HTTP error! status: 500');
  });

  it('throws default message on HTTP error with empty body', async () => {
    const resp = {
      ok: false,
      status: 502,
      text: jest.fn().mockResolvedValue(''),
      headers: new Headers(),
    } as unknown as Response;
    mockFetch.mockResolvedValueOnce(resp);
    await expect(pyreflectAPI.healthCheck()).rejects.toThrow('HTTP error! status: 502');
  });

  it('throws on network error', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'));
    await expect(pyreflectAPI.healthCheck()).rejects.toThrow('Network error');
  });

  it('returns undefined for void (empty body) responses', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse('', 204, true));
    // deleteHistoryItem returns void
    const res = await pyreflectAPI.deleteHistoryItem('abc');
    expect(res).toBeUndefined();
  });

  it('merges custom headers without losing Content-Type', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse(undefined, 204, true));
    await pyreflectAPI.forcePurgeJob('j1', 'tok123');
    const [, opts] = mockFetch.mock.calls[0];
    expect(opts.headers['Content-Type']).toBe('application/json');
    expect(opts.headers['X-Admin-Token']).toBe('tok123');
  });
});

// ================================================================
// CORE ENDPOINTS
// ================================================================
describe('Core endpoints', () => {
  it('getLimits()', async () => {
    const body = { production: true, limits: {} };
    mockFetch.mockResolvedValueOnce(mockResponse(body));
    expect(await pyreflectAPI.getLimits()).toEqual(body);
  });

  it('getDefaults()', async () => {
    const body = { layers: [], generator: {}, training: {} };
    mockFetch.mockResolvedValueOnce(mockResponse(body));
    expect(await pyreflectAPI.getDefaults()).toEqual(body);
  });

  it('getStatus()', async () => {
    const body = { status: 'running', data_files: ['a.npy'] };
    mockFetch.mockResolvedValueOnce(mockResponse(body));
    expect(await pyreflectAPI.getStatus()).toEqual(body);
  });
});

// ================================================================
// GENERATION ENDPOINTS
// ================================================================
describe('Generation endpoints', () => {
  const params = { layers: [], generator: {} as any, training: {} as any };

  it('generateData() sends POST and returns response', async () => {
    const body = { nr: {}, sld: {}, training: {}, chi: [], metrics: {} };
    mockFetch.mockResolvedValueOnce(mockResponse(body));
    expect(await pyreflectAPI.generateData(params)).toEqual(body);
    expect(mockFetch.mock.calls[0][1].method).toBe('POST');
  });

  describe('generateStream()', () => {
    it('returns complete event and calls onProgress', async () => {
      const lines = [
        'data: {"type":"progress","message":"step1"}',
        'data: {"type":"progress","message":"step2"}',
        'data: {"type":"complete","result":"done"}',
      ];
      mockFetch.mockResolvedValueOnce(mockSSEResponse(lines));

      const progress: any[] = [];
      const result = await pyreflectAPI.generateStream(params, (e) => progress.push(e));

      expect(progress).toHaveLength(2);
      expect(progress[0].message).toBe('step1');
      expect(result).toEqual({ type: 'complete', result: 'done' });
    });

    it('returns null when no complete event is received', async () => {
      const lines = ['data: {"type":"progress","message":"step1"}'];
      mockFetch.mockResolvedValueOnce(mockSSEResponse(lines));
      const result = await pyreflectAPI.generateStream(params, null);
      expect(result).toBeNull();
    });

    it('throws on non-ok response', async () => {
      mockFetch.mockResolvedValueOnce(mockSSEResponse([], 500));
      await expect(pyreflectAPI.generateStream(params)).rejects.toThrow('HTTP error! status: 500');
    });

    it('handles malformed SSE data gracefully', async () => {
      const lines = [
        'data: {invalid json}',
        'data: {"type":"complete","ok":true}',
      ];
      mockFetch.mockResolvedValueOnce(mockSSEResponse(lines));
      const result = await pyreflectAPI.generateStream(params);
      expect(result).toEqual({ type: 'complete', ok: true });
    });

    it('skips non-data lines', async () => {
      const lines = [
        ': keep-alive',
        'event: ping',
        'data: {"type":"complete"}',
      ];
      mockFetch.mockResolvedValueOnce(mockSSEResponse(lines));
      const result = await pyreflectAPI.generateStream(params);
      expect(result).toEqual({ type: 'complete' });
    });

    it('works with onProgress = undefined', async () => {
      const lines = ['data: {"type":"progress","message":"x"}'];
      mockFetch.mockResolvedValueOnce(mockSSEResponse(lines));
      const result = await pyreflectAPI.generateStream(params);
      expect(result).toBeNull();
    });
  });
});

// ================================================================
// HISTORY ENDPOINTS
// ================================================================
describe('History endpoints', () => {
  it('getHistory()', async () => {
    const body = { items: [{ id: '1', name: 'run1', created_at: '2025-01-01' }] };
    mockFetch.mockResolvedValueOnce(mockResponse(body));
    expect(await pyreflectAPI.getHistory()).toEqual(body);
  });

  it('saveGeneration()', async () => {
    const data = { name: 'test', result: {} as any, config: {} };
    const entry = { id: '2', name: 'test', created_at: '2025-01-01' };
    mockFetch.mockResolvedValueOnce(mockResponse(entry));
    expect(await pyreflectAPI.saveGeneration(data)).toEqual(entry);
    expect(mockFetch.mock.calls[0][1].method).toBe('POST');
  });

  it('getHistoryItem()', async () => {
    const entry = { id: '3', name: 'run3', created_at: '2025-01-01' };
    mockFetch.mockResolvedValueOnce(mockResponse(entry));
    expect(await pyreflectAPI.getHistoryItem('3')).toEqual(entry);
    expect(mockFetch.mock.calls[0][0]).toContain('/api/history/3');
  });

  it('renameHistoryItem()', async () => {
    const entry = { id: '4', name: 'new', created_at: '2025-01-01' };
    mockFetch.mockResolvedValueOnce(mockResponse(entry));
    expect(await pyreflectAPI.renameHistoryItem('4', 'new')).toEqual(entry);
    expect(mockFetch.mock.calls[0][1].method).toBe('PATCH');
  });

  it('deleteHistoryItem()', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse('', 204, true));
    await expect(pyreflectAPI.deleteHistoryItem('5')).resolves.toBeUndefined();
    expect(mockFetch.mock.calls[0][1].method).toBe('DELETE');
  });
});

// ================================================================
// MODELS ENDPOINTS
// ================================================================
describe('Models endpoints', () => {
  it('uploadModel()', async () => {
    const body = { model_id: 'm1', size_mb: 50 };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: jest.fn().mockResolvedValue(body),
    } as unknown as Response);
    const form = new FormData();
    expect(await pyreflectAPI.uploadModel(form)).toEqual(body);
    expect(mockFetch.mock.calls[0][1].method).toBe('POST');
  });

  it('uploadModel() throws on error', async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 413 } as unknown as Response);
    await expect(pyreflectAPI.uploadModel(new FormData())).rejects.toThrow('HTTP error! status: 413');
  });

  it('downloadModel()', async () => {
    const blob = new Blob(['test']);
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      blob: jest.fn().mockResolvedValue(blob),
    } as unknown as Response);
    const res = await pyreflectAPI.downloadModel('m1');
    expect(res).toBe(blob);
  });

  it('downloadModel() throws on error', async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 404 } as unknown as Response);
    await expect(pyreflectAPI.downloadModel('m1')).rejects.toThrow('HTTP error! status: 404');
  });

  it('deleteModel()', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse('', 204, true));
    await expect(pyreflectAPI.deleteModel('m1')).resolves.toBeUndefined();
  });

  it('getModelInfo()', async () => {
    const body = { model_id: 'm1', size_mb: 10, source: 'upload' };
    mockFetch.mockResolvedValueOnce(mockResponse(body));
    expect(await pyreflectAPI.getModelInfo('m1')).toEqual(body);
  });

  describe('uploadFiles()', () => {
    it('uploads files with roles', async () => {
      const body = { model_id: 'upload1' };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: jest.fn().mockResolvedValue(body),
      } as unknown as Response);
      const file = new File(['content'], 'test.npy');
      expect(await pyreflectAPI.uploadFiles([file], ['nr_train'])).toEqual(body);
    });

    it('uploads files without roles', async () => {
      const body = { model_id: 'upload2' };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: jest.fn().mockResolvedValue(body),
      } as unknown as Response);
      const file = new File(['content'], 'test.npy');
      expect(await pyreflectAPI.uploadFiles([file])).toEqual(body);
    });

    it('uploads files with null roles', async () => {
      const body = { model_id: 'upload3' };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: jest.fn().mockResolvedValue(body),
      } as unknown as Response);
      const file = new File(['content'], 'test.npy');
      expect(await pyreflectAPI.uploadFiles([file], null)).toEqual(body);
    });

    it('throws on error', async () => {
      mockFetch.mockResolvedValueOnce({ ok: false, status: 500 } as unknown as Response);
      await expect(pyreflectAPI.uploadFiles([new File(['x'], 'a.npy')])).rejects.toThrow(
        'HTTP error! status: 500'
      );
    });
  });
});

// ================================================================
// JOBS ENDPOINTS
// ================================================================
describe('Jobs endpoints', () => {
  const jobResp = { job_id: 'j1', status: 'queued' as const };

  it('submitJob()', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse(jobResp));
    expect(await pyreflectAPI.submitJob({ type: 'train' })).toEqual(jobResp);
    expect(mockFetch.mock.calls[0][1].method).toBe('POST');
  });

  it('getJobStatus()', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse(jobResp));
    expect(await pyreflectAPI.getJobStatus('j1')).toEqual(jobResp);
  });

  it('cancelJob()', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse('', 204, true));
    await expect(pyreflectAPI.cancelJob('j1')).resolves.toBeUndefined();
    expect(mockFetch.mock.calls[0][1].method).toBe('DELETE');
  });

  it('renameJob()', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse({ ...jobResp, name: 'new' }));
    const res = await pyreflectAPI.renameJob('j1', 'new');
    expect(res.name).toBe('new');
    expect(mockFetch.mock.calls[0][1].method).toBe('PATCH');
  });

  it('retryJob()', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse(jobResp));
    expect(await pyreflectAPI.retryJob('j1')).toEqual(jobResp);
    expect(mockFetch.mock.calls[0][1].method).toBe('POST');
  });

  it('stopJob()', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse('', 204, true));
    await expect(pyreflectAPI.stopJob('j1')).resolves.toBeUndefined();
  });

  it('pauseJob()', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse('', 204, true));
    await expect(pyreflectAPI.pauseJob('j1')).resolves.toBeUndefined();
  });

  it('deleteJob()', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse('', 204, true));
    await expect(pyreflectAPI.deleteJob('j1')).resolves.toBeUndefined();
    expect(mockFetch.mock.calls[0][1].method).toBe('DELETE');
  });

  it('claimJob()', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse(jobResp));
    expect(await pyreflectAPI.claimJob('j1', 'u1')).toEqual(jobResp);
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.userId).toBe('u1');
  });

  it('purgeUserJobs()', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse('', 204, true));
    await expect(pyreflectAPI.purgeUserJobs('u1')).resolves.toBeUndefined();
  });

  it('forcePurgeJob()', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse('', 204, true));
    await expect(pyreflectAPI.forcePurgeJob('j1', 'admin-tok')).resolves.toBeUndefined();
    const [, opts] = mockFetch.mock.calls[0];
    expect(opts.headers['X-Admin-Token']).toBe('admin-tok');
  });
});

// ================================================================
// CHECKPOINTS ENDPOINTS
// ================================================================
describe('Checkpoints endpoints', () => {
  it('getCheckpoints()', async () => {
    const body = { checkpoints: [{ job_id: 'j1', created_at: '2025-01-01' }] };
    mockFetch.mockResolvedValueOnce(mockResponse(body));
    expect(await pyreflectAPI.getCheckpoints()).toEqual(body);
  });

  it('resumeFromCheckpoint()', async () => {
    const body = { job_id: 'j1', status: 'running' };
    mockFetch.mockResolvedValueOnce(mockResponse(body));
    expect(await pyreflectAPI.resumeFromCheckpoint('j1')).toEqual(body);
    expect(mockFetch.mock.calls[0][1].method).toBe('POST');
  });

  it('deleteCheckpoint()', async () => {
    mockFetch.mockResolvedValueOnce(mockResponse('', 204, true));
    await expect(pyreflectAPI.deleteCheckpoint('j1')).resolves.toBeUndefined();
  });
});

// ================================================================
// QUEUE ENDPOINTS
// ================================================================
describe('Queue endpoints', () => {
  it('getQueueStatus()', async () => {
    const body = { queue_length: 3, workers: 2, active_jobs: 1 };
    mockFetch.mockResolvedValueOnce(mockResponse(body));
    expect(await pyreflectAPI.getQueueStatus()).toEqual(body);
  });

  it('spawnWorker()', async () => {
    const body = { success: true };
    mockFetch.mockResolvedValueOnce(mockResponse(body));
    expect(await pyreflectAPI.spawnWorker()).toEqual(body);
  });

  it('cleanupQueue() without dry_run', async () => {
    const body = { deleted: 5, dry_run: false };
    mockFetch.mockResolvedValueOnce(mockResponse(body));
    expect(await pyreflectAPI.cleanupQueue('tok')).toEqual(body);
    expect(mockFetch.mock.calls[0][0]).not.toContain('dry_run');
    expect(mockFetch.mock.calls[0][1].headers['X-Admin-Token']).toBe('tok');
  });

  it('cleanupQueue() with dry_run', async () => {
    const body = { deleted: 0, dry_run: true };
    mockFetch.mockResolvedValueOnce(mockResponse(body));
    expect(await pyreflectAPI.cleanupQueue('tok', true)).toEqual(body);
    expect(mockFetch.mock.calls[0][0]).toContain('?dry_run=true');
  });
});

// ================================================================
// UTILITY METHODS
// ================================================================

/** Real-timer delay for fire-and-forget async chain draining */
const delay = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/** Flush microtask queue (for use alongside fake timers). */
const flushMicrotasks = () => new Promise<void>((r) => process.nextTick(r));

describe('Utility methods', () => {
  describe('pollJobStatus()', () => {
    it('returns immediately on completed', async () => {
      const completed = { job_id: 'j1', status: 'completed' as const };
      mockFetch.mockResolvedValueOnce(mockResponse(completed));

      const updates: any[] = [];
      const result = await pyreflectAPI.pollJobStatus('j1', (s) => updates.push(s));
      expect(result.status).toBe('completed');
      expect(updates).toHaveLength(1);
    });

    it('throws on failed job', async () => {
      const failed = { job_id: 'j1', status: 'failed' as const, error: 'OOM' };
      mockFetch.mockResolvedValueOnce(mockResponse(failed));

      await expect(pyreflectAPI.pollJobStatus('j1')).rejects.toThrow('OOM');
    });

    it('throws default message on failed job without error', async () => {
      const failed = { job_id: 'j1', status: 'failed' as const };
      mockFetch.mockResolvedValueOnce(mockResponse(failed));

      await expect(pyreflectAPI.pollJobStatus('j1')).rejects.toThrow('Job failed');
    });

    it('re-throws network errors during polling', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Connection reset'));
      await expect(pyreflectAPI.pollJobStatus('j1')).rejects.toThrow('Connection reset');
    });

    it('polls until completed', async () => {
      // Use real timers but mock setTimeout to be fast
      const origSetTimeout = globalThis.setTimeout;
      (globalThis as any).setTimeout = (fn: () => void) => origSetTimeout(fn, 0);

      const running = { job_id: 'j1', status: 'running' as const };
      const completed = { job_id: 'j1', status: 'completed' as const };

      mockFetch
        .mockResolvedValueOnce(mockResponse(running))
        .mockResolvedValueOnce(mockResponse(completed));

      const result = await pyreflectAPI.pollJobStatus('j1', null, 5);
      expect(result.status).toBe('completed');

      globalThis.setTimeout = origSetTimeout;
    });

    it('times out after maxAttempts', async () => {
      const origSetTimeout = globalThis.setTimeout;
      (globalThis as any).setTimeout = (fn: () => void) => origSetTimeout(fn, 0);

      const running = { job_id: 'j1', status: 'running' as const };
      mockFetch.mockResolvedValue(mockResponse(running));

      await expect(pyreflectAPI.pollJobStatus('j1', null, 1)).rejects.toThrow(
        'Job polling timeout'
      );

      globalThis.setTimeout = origSetTimeout;
    });
  });

  describe('streamJobProgress()', () => {
    it('calls onComplete on completed job', async () => {
      const completed = { job_id: 'j1', status: 'completed' as const };
      mockFetch.mockResolvedValueOnce(mockResponse(completed));

      const onProgress = jest.fn();
      const onComplete = jest.fn();
      const onError = jest.fn();

      pyreflectAPI.streamJobProgress('j1', onProgress, onComplete, onError);

      // Fire-and-forget poll: wait for async chain to resolve
      await delay(50);

      expect(onProgress).toHaveBeenCalledWith(completed);
      expect(onComplete).toHaveBeenCalledWith(completed);
      expect(onError).not.toHaveBeenCalled();
    });

    it('calls onError on failed job', async () => {
      const failed = { job_id: 'j1', status: 'failed' as const, error: 'OOM' };
      mockFetch.mockResolvedValueOnce(mockResponse(failed));

      const onError = jest.fn();
      pyreflectAPI.streamJobProgress('j1', undefined, undefined, onError);
      await delay(50);

      expect(onError).toHaveBeenCalledWith('OOM');
    });

    it('calls onError with default message on failed job without error', async () => {
      const failed = { job_id: 'j1', status: 'failed' as const };
      mockFetch.mockResolvedValueOnce(mockResponse(failed));

      const onError = jest.fn();
      pyreflectAPI.streamJobProgress('j1', undefined, undefined, onError);
      await delay(50);

      expect(onError).toHaveBeenCalledWith('Job failed');
    });

    it('calls onError on network failure with Error object', async () => {
      mockFetch.mockRejectedValueOnce(new Error('timeout'));

      const onError = jest.fn();
      pyreflectAPI.streamJobProgress('j1', undefined, undefined, onError);
      await delay(50);

      expect(onError).toHaveBeenCalledWith(expect.any(Error));
    });

    it('calls onError on network failure with non-Error', async () => {
      mockFetch.mockRejectedValueOnce('string error');

      const onError = jest.fn();
      pyreflectAPI.streamJobProgress('j1', undefined, undefined, onError);
      await delay(50);

      expect(onError).toHaveBeenCalledWith('string error');
    });

    it('continues polling on running job', async () => {
      const running = { job_id: 'j1', status: 'running' as const };
      const completed = { job_id: 'j1', status: 'completed' as const };

      mockFetch
        .mockResolvedValueOnce(mockResponse(running))
        .mockResolvedValueOnce(mockResponse(completed));

      const onComplete = jest.fn();
      pyreflectAPI.streamJobProgress('j1', undefined, onComplete);

      // Wait for first poll + 1000ms setTimeout + second poll to complete
      await delay(1200);

      expect(onComplete).toHaveBeenCalledWith(completed);
    }, 3000);

    it('works without any callbacks', async () => {
      const completed = { job_id: 'j1', status: 'completed' as const };
      mockFetch.mockResolvedValueOnce(mockResponse(completed));

      // Should not throw
      pyreflectAPI.streamJobProgress('j1');
      await delay(50);
    });
  });

  describe('batchDeleteJobs()', () => {
    it('settles all promises', async () => {
      mockFetch
        .mockResolvedValueOnce(mockResponse('', 204, true))
        .mockRejectedValueOnce(new Error('fail'));

      const results = await pyreflectAPI.batchDeleteJobs(['j1', 'j2']);
      expect(results).toHaveLength(2);
      expect(results[0].status).toBe('fulfilled');
      expect(results[1].status).toBe('rejected');
    });
  });

  describe('batchRetryJobs()', () => {
    it('settles all promises', async () => {
      const resp = { job_id: 'j1', status: 'queued' };
      mockFetch
        .mockResolvedValueOnce(mockResponse(resp))
        .mockResolvedValueOnce(mockResponse(resp));

      const results = await pyreflectAPI.batchRetryJobs(['j1', 'j2']);
      expect(results).toHaveLength(2);
      expect(results.every((r) => r.status === 'fulfilled')).toBe(true);
    });
  });
});

// ================================================================
// MODULE EXPORTS
// ================================================================
describe('Module exports', () => {
  it('exports named pyreflectAPI instance', () => {
    expect(pyreflectAPI).toBeDefined();
    expect(typeof pyreflectAPI.healthCheck).toBe('function');
  });

  it('exports default', async () => {
    const mod = await import('./pyreflectAPI');
    expect(mod.default).toBe(mod.pyreflectAPI);
  });
});
