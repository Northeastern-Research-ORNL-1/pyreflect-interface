// Complete PyReflect API Implementation
// File: src/interface/src/services/pyreflectAPI.ts

import type {
  // Core
  HealthResponse,
  DefaultsResponse,
  StatusResponse,
  LimitsResponse,
  // Generation
  GenerateParams,
  GenerateResponse,
  StreamProgressEvent,
  StreamCompleteEvent,
  // History
  HistoryListResponse,
  HistoryEntry,
  SaveGenerationData,
  // Models
  ModelUploadResponse,
  ModelInfoResponse,
  // Jobs
  JobStatusResponse,
  JobSubmitData,
  // Checkpoints
  CheckpointsListResponse,
  // Queue
  QueueStatusResponse,
  SpawnWorkerResponse,
  CleanupQueueResponse,
  // Error
  ApiErrorResponse,
} from '@/types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class PyReflectAPI {
  private readonly baseURL: string;

  constructor() {
    this.baseURL = API_BASE;
  }

  // Helper method for making requests
  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    const { headers: optionHeaders, ...restOptions } = options;
    const config: RequestInit = {
      ...restOptions,
      headers: {
        'Content-Type': 'application/json',
        ...(optionHeaders as Record<string, string>),
      },
    };

    try {
      const response = await fetch(url, config);

      // Handle empty responses (204 No Content, etc.)
      const text = await response.text();

      if (!response.ok) {
        let errorMessage = `HTTP error! status: ${response.status}`;
        if (text) {
          try {
            const errorData = JSON.parse(text) as ApiErrorResponse;
            errorMessage = errorData.message ?? errorData.detail ?? errorMessage;
          } catch {
            // Response is not JSON, use default error message
          }
        }
        throw new Error(errorMessage);
      }

      // void responses (e.g. DELETE 204)
      if (!text) {
        return undefined as T;
      }

      return JSON.parse(text) as T;
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  // ========================================
  // CORE ENDPOINTS
  // ========================================

  // GET /api/health - Health check
  async healthCheck(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/api/health');
  }

  // GET /api/limits - Current limits + access status
  async getLimits(): Promise<LimitsResponse> {
    return this.request<LimitsResponse>('/api/limits');
  }

  // GET /api/defaults - Default parameters
  async getDefaults(): Promise<DefaultsResponse> {
    return this.request<DefaultsResponse>('/api/defaults');
  }

  // GET /api/status - Backend status and data files
  async getStatus(): Promise<StatusResponse> {
    return this.request<StatusResponse>('/api/status');
  }

  // ========================================
  // GENERATION ENDPOINTS
  // ========================================

  // POST /api/generate - Generate NR/SLD curves (non-streaming)
  async generateData(parameters: GenerateParams): Promise<GenerateResponse> {
    return this.request<GenerateResponse>('/api/generate', {
      method: 'POST',
      body: JSON.stringify(parameters),
    });
  }

  // POST /api/generate/stream - Generate with SSE log stream
  async generateStream(
    parameters: GenerateParams,
    onProgress?: ((event: StreamProgressEvent) => void) | null
  ): Promise<StreamCompleteEvent | null> {
    const url = `${this.baseURL}/api/generate/stream`;

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify(parameters),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Handle Server-Sent Events (SSE)
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();

    let buffer = '';
    let finalResult: StreamCompleteEvent | null = null;

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? ''; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6)) as { type: string } & Record<string, unknown>;

            if (data.type === 'progress' && onProgress) {
              onProgress(data as StreamProgressEvent);
            } else if (data.type === 'complete') {
              finalResult = data as StreamCompleteEvent;
            }
          } catch (e) {
            console.warn('Failed to parse SSE data:', line);
          }
        }
      }
    }

    return finalResult;
  }

  // ========================================
  // HISTORY ENDPOINTS
  // ========================================

  // GET /api/history - List saved generations
  async getHistory(): Promise<HistoryListResponse> {
    return this.request<HistoryListResponse>('/api/history');
  }

  // POST /api/history - Save a generation manually
  async saveGeneration(data: SaveGenerationData): Promise<HistoryEntry> {
    return this.request<HistoryEntry>('/api/history', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // GET /api/history/{id} - Get full details of a save
  async getHistoryItem(id: string): Promise<HistoryEntry> {
    return this.request<HistoryEntry>(`/api/history/${id}`);
  }

  // PATCH /api/history/{id} - Rename a saved generation
  async renameHistoryItem(id: string, newName: string): Promise<HistoryEntry> {
    return this.request<HistoryEntry>(`/api/history/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ name: newName }),
    });
  }

  // DELETE /api/history/{id} - Delete a saved generation and its model
  async deleteHistoryItem(id: string): Promise<void> {
    return this.request<void>(`/api/history/${id}`, {
      method: 'DELETE',
    });
  }

  // ========================================
  // MODELS ENDPOINTS
  // ========================================

  // POST /api/models/upload - Receive model upload from worker
  async uploadModel(formData: FormData): Promise<ModelUploadResponse> {
    const url = `${this.baseURL}/api/models/upload`;

    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json() as Promise<ModelUploadResponse>;
  }

  // GET /api/models/{model_id} - Download a saved model
  async downloadModel(modelId: string): Promise<Blob> {
    const url = `${this.baseURL}/api/models/${modelId}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.blob();
  }

  // DELETE /api/models/{model_id} - Delete a local model file
  async deleteModel(modelId: string): Promise<void> {
    return this.request<void>(`/api/models/${modelId}`, {
      method: 'DELETE',
    });
  }

  // GET /api/models/{model_id}/info - Get model size and source
  async getModelInfo(modelId: string): Promise<ModelInfoResponse> {
    return this.request<ModelInfoResponse>(`/api/models/${modelId}/info`);
  }

  // POST /api/upload - Upload files (+ optional roles)
  async uploadFiles(files: File[], roles?: string[] | null): Promise<ModelUploadResponse> {
    const formData = new FormData();

    files.forEach((file, index) => {
      formData.append('files', file);
      if (roles && roles[index]) {
        formData.append(`roles[${index}]`, roles[index]);
      }
    });

    const url = `${this.baseURL}/api/upload`;
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json() as Promise<ModelUploadResponse>;
  }

  // ========================================
  // JOBS ENDPOINTS
  // ========================================

  // POST /api/jobs/submit - Submit job to queue (non-blocking)
  async submitJob(jobData: JobSubmitData): Promise<JobStatusResponse> {
    return this.request<JobStatusResponse>('/api/jobs/submit', {
      method: 'POST',
      body: JSON.stringify(jobData),
    });
  }

  // GET /api/jobs/{job_id} - Get job status, progress, and result
  async getJobStatus(jobId: string): Promise<JobStatusResponse> {
    return this.request<JobStatusResponse>(`/api/jobs/${jobId}`);
  }

  // DELETE /api/jobs/{job_id} - Cancel a queued job
  async cancelJob(jobId: string): Promise<void> {
    return this.request<void>(`/api/jobs/${jobId}`, {
      method: 'DELETE',
    });
  }

  // PATCH /api/jobs/{job_id}/name - Rename a queued job
  async renameJob(jobId: string, newName: string): Promise<JobStatusResponse> {
    return this.request<JobStatusResponse>(`/api/jobs/${jobId}/name`, {
      method: 'PATCH',
      body: JSON.stringify({ name: newName }),
    });
  }

  // POST /api/jobs/{job_id}/retry - Retry a failed/finished job
  async retryJob(jobId: string): Promise<JobStatusResponse> {
    return this.request<JobStatusResponse>(`/api/jobs/${jobId}/retry`, {
      method: 'POST',
    });
  }

  // POST /api/jobs/{job_id}/stop - Stop job immediately (no checkpoint)
  async stopJob(jobId: string): Promise<void> {
    return this.request<void>(`/api/jobs/${jobId}/stop`, {
      method: 'POST',
    });
  }

  // POST /api/jobs/{job_id}/pause - Pause job and save checkpoint
  async pauseJob(jobId: string): Promise<void> {
    return this.request<void>(`/api/jobs/${jobId}/pause`, {
      method: 'POST',
    });
  }

  // DELETE /api/jobs/{job_id}/delete - Delete a job record (non-running only)
  async deleteJob(jobId: string): Promise<void> {
    return this.request<void>(`/api/jobs/${jobId}/delete`, {
      method: 'DELETE',
    });
  }

  // POST /api/jobs/{job_id}/claim - Attach a job to a user (login mid-run)
  async claimJob(jobId: string, userId: string): Promise<JobStatusResponse> {
    return this.request<JobStatusResponse>(`/api/jobs/${jobId}/claim`, {
      method: 'POST',
      body: JSON.stringify({ userId }),
    });
  }

  // DELETE /api/jobs/purge - Delete non-running jobs for a user
  async purgeUserJobs(userId: string): Promise<void> {
    return this.request<void>('/api/jobs/purge', {
      method: 'DELETE',
      body: JSON.stringify({ userId }),
    });
  }

  // POST /api/jobs/{job_id}/force-purge - Force purge a zombie job (admin)
  async forcePurgeJob(jobId: string, adminToken: string): Promise<void> {
    return this.request<void>(`/api/jobs/${jobId}/force-purge`, {
      method: 'POST',
      headers: {
        'X-Admin-Token': adminToken,
      },
    });
  }

  // ========================================
  // CHECKPOINTS ENDPOINTS
  // ========================================

  // GET /api/checkpoints - List all available checkpoints
  async getCheckpoints(): Promise<CheckpointsListResponse> {
    return this.request<CheckpointsListResponse>('/api/checkpoints');
  }

  // POST /api/checkpoints/{job_id}/resume - Resume training from checkpoint
  async resumeFromCheckpoint(jobId: string): Promise<JobStatusResponse> {
    return this.request<JobStatusResponse>(`/api/checkpoints/${jobId}/resume`, {
      method: 'POST',
    });
  }

  // DELETE /api/checkpoints/{job_id} - Delete a checkpoint
  async deleteCheckpoint(jobId: string): Promise<void> {
    return this.request<void>(`/api/checkpoints/${jobId}`, {
      method: 'DELETE',
    });
  }

  // ========================================
  // QUEUE ENDPOINTS
  // ========================================

  // GET /api/queue - Queue status and worker info
  async getQueueStatus(): Promise<QueueStatusResponse> {
    return this.request<QueueStatusResponse>('/api/queue');
  }

  // POST /api/queue/spawn - Trigger remote worker spawn (debug)
  async spawnWorker(): Promise<SpawnWorkerResponse> {
    return this.request<SpawnWorkerResponse>('/api/queue/spawn', {
      method: 'POST',
    });
  }

  // POST /api/queue/cleanup - Trigger stale job cleanup (admin)
  async cleanupQueue(adminToken: string, dryRun = false): Promise<CleanupQueueResponse> {
    const params = dryRun ? '?dry_run=true' : '';
    return this.request<CleanupQueueResponse>(`/api/queue/cleanup${params}`, {
      method: 'POST',
      headers: {
        'X-Admin-Token': adminToken,
      },
    });
  }

  // ========================================
  // UTILITY METHODS
  // ========================================

  // Poll job status until completion
  async pollJobStatus(
    jobId: string,
    onUpdate?: ((status: JobStatusResponse) => void) | null,
    maxAttempts = 120
  ): Promise<JobStatusResponse> {
    let attempts = 0;

    const poll = async (): Promise<JobStatusResponse> => {
      try {
        const status = await this.getJobStatus(jobId);

        if (onUpdate) {
          onUpdate(status);
        }

        if (status.status === 'completed') {
          return status;
        } else if (status.status === 'failed') {
          throw new Error(status.error ?? 'Job failed');
        } else if (attempts >= maxAttempts) {
          throw new Error('Job polling timeout');
        } else {
          attempts++;
          await new Promise<void>((resolve) => setTimeout(resolve, 2000));
          return poll();
        }
      } catch (error) {
        console.error('Job polling error:', error);
        throw error;
      }
    };

    return poll();
  }

  // Stream job progress with real-time updates
  async streamJobProgress(
    jobId: string,
    onProgress?: (status: JobStatusResponse) => void,
    onComplete?: (status: JobStatusResponse) => void,
    onError?: (error: string | Error) => void
  ): Promise<void> {
    const poll = async (): Promise<void> => {
      try {
        const status = await this.getJobStatus(jobId);

        if (onProgress) {
          onProgress(status);
        }

        if (status.status === 'completed') {
          if (onComplete) onComplete(status);
          return;
        } else if (status.status === 'failed') {
          if (onError) onError(status.error ?? 'Job failed');
          return;
        }

        setTimeout(poll, 1000); // Poll every second
      } catch (error) {
        if (onError) onError(error instanceof Error ? error : String(error));
      }
    };

    poll();
  }

  // Batch operations
  async batchDeleteJobs(jobIds: string[]): Promise<PromiseSettledResult<void>[]> {
    const promises = jobIds.map((id) => this.deleteJob(id));
    return Promise.allSettled(promises);
  }

  async batchRetryJobs(jobIds: string[]): Promise<PromiseSettledResult<JobStatusResponse>[]> {
    const promises = jobIds.map((id) => this.retryJob(id));
    return Promise.allSettled(promises);
  }
}

// Export singleton instance
export const pyreflectAPI = new PyReflectAPI();
export default pyreflectAPI;
