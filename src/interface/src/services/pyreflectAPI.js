// Complete PyReflect API Implementation
// File: src/interface/src/services/pyreflectAPI.js

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class PyReflectAPI {
  constructor() {
    this.baseURL = API_BASE;
  }

  // Helper method for making requests
  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.message || `HTTP error! status: ${response.status}`);
      }
      
      return data;
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  // ========================================
  // CORE ENDPOINTS
  // ========================================

  // GET /api/health - Health check
  async healthCheck() {
    return this.request('/api/health');
  }

  // GET /api/limits - Current limits + access status
  async getLimits() {
    return this.request('/api/limits');
  }

  // GET /api/defaults - Default parameters
  async getDefaults() {
    return this.request('/api/defaults');
  }

  // GET /api/status - Backend status and data files
  async getStatus() {
    return this.request('/api/status');
  }

  // ========================================
  // GENERATION ENDPOINTS
  // ========================================

  // POST /api/generate - Generate NR/SLD curves (non-streaming)
  async generateData(parameters) {
    return this.request('/api/generate', {
      method: 'POST',
      body: JSON.stringify(parameters),
    });
  }

  // POST /api/generate/stream - Generate with SSE log stream
  async generateStream(parameters, onProgress = null) {
    const url = `${this.baseURL}/api/generate/stream`;
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify(parameters),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Handle Server-Sent Events (SSE)
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let buffer = '';
    let finalResult = null;

    while (true) {
      const { done, value } = await reader.read();
      
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            
            if (data.type === 'progress' && onProgress) {
              onProgress(data);
            } else if (data.type === 'complete') {
              finalResult = data;
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
  async getHistory() {
    return this.request('/api/history');
  }

  // POST /api/history - Save a generation manually
  async saveGeneration(data) {
    return this.request('/api/history', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // GET /api/history/{id} - Get full details of a save
  async getHistoryItem(id) {
    return this.request(`/api/history/${id}`);
  }

  // PATCH /api/history/{id} - Rename a saved generation
  async renameHistoryItem(id, newName) {
    return this.request(`/api/history/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ name: newName }),
    });
  }

  // DELETE /api/history/{id} - Delete a saved generation and its model
  async deleteHistoryItem(id) {
    return this.request(`/api/history/${id}`, {
      method: 'DELETE',
    });
  }

  // ========================================
  // MODELS ENDPOINTS
  // ========================================

  // POST /api/models/upload - Receive model upload from worker
  async uploadModel(formData) {
    const url = `${this.baseURL}/api/models/upload`;
    
    const response = await fetch(url, {
      method: 'POST',
      body: formData, // FormData object
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  // GET /api/models/{model_id} - Download a saved model
  async downloadModel(modelId) {
    const url = `${this.baseURL}/api/models/${modelId}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.blob();
  }

  // DELETE /api/models/{model_id} - Delete a local model file
  async deleteModel(modelId) {
    return this.request(`/api/models/${modelId}`, {
      method: 'DELETE',
    });
  }

  // GET /api/models/{model_id}/info - Get model size and source
  async getModelInfo(modelId) {
    return this.request(`/api/models/${modelId}/info`);
  }

  // POST /api/upload - Upload files (+ optional roles)
  async uploadFiles(files, roles = null) {
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

    return response.json();
  }

  // ========================================
  // JOBS ENDPOINTS
  // ========================================

  // POST /api/jobs/submit - Submit job to queue (non-blocking)
  async submitJob(jobData) {
    return this.request('/api/jobs/submit', {
      method: 'POST',
      body: JSON.stringify(jobData),
    });
  }

  // GET /api/jobs/{job_id} - Get job status, progress, and result
  async getJobStatus(jobId) {
    return this.request(`/api/jobs/${jobId}`);
  }

  // DELETE /api/jobs/{job_id} - Cancel a queued job
  async cancelJob(jobId) {
    return this.request(`/api/jobs/${jobId}`, {
      method: 'DELETE',
    });
  }

  // PATCH /api/jobs/{job_id}/name - Rename a queued job
  async renameJob(jobId, newName) {
    return this.request(`/api/jobs/${jobId}/name`, {
      method: 'PATCH',
      body: JSON.stringify({ name: newName }),
    });
  }

  // POST /api/jobs/{job_id}/retry - Retry a failed/finished job
  async retryJob(jobId) {
    return this.request(`/api/jobs/${jobId}/retry`, {
      method: 'POST',
    });
  }

  // POST /api/jobs/{job_id}/stop - Stop job immediately (no checkpoint)
  async stopJob(jobId) {
    return this.request(`/api/jobs/${jobId}/stop`, {
      method: 'POST',
    });
  }

  // POST /api/jobs/{job_id}/pause - Pause job and save checkpoint
  async pauseJob(jobId) {
    return this.request(`/api/jobs/${jobId}/pause`, {
      method: 'POST',
    });
  }

  // DELETE /api/jobs/{job_id}/delete - Delete a job record (non-running only)
  async deleteJob(jobId) {
    return this.request(`/api/jobs/${jobId}/delete`, {
      method: 'DELETE',
    });
  }

  // POST /api/jobs/{job_id}/claim - Attach a job to a user (login mid-run)
  async claimJob(jobId, userId) {
    return this.request(`/api/jobs/${jobId}/claim`, {
      method: 'POST',
      body: JSON.stringify({ userId }),
    });
  }

  // DELETE /api/jobs/purge - Delete non-running jobs for a user
  async purgeUserJobs(userId) {
    return this.request('/api/jobs/purge', {
      method: 'DELETE',
      body: JSON.stringify({ userId }),
    });
  }

  // POST /api/jobs/{job_id}/force-purge - Force purge a zombie job (admin)
  async forcePurgeJob(jobId, adminToken) {
    return this.request(`/api/jobs/${jobId}/force-purge`, {
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
  async getCheckpoints() {
    return this.request('/api/checkpoints');
  }

  // POST /api/checkpoints/{job_id}/resume - Resume training from checkpoint
  async resumeFromCheckpoint(jobId) {
    return this.request(`/api/checkpoints/${jobId}/resume`, {
      method: 'POST',
    });
  }

  // DELETE /api/checkpoints/{job_id} - Delete a checkpoint
  async deleteCheckpoint(jobId) {
    return this.request(`/api/checkpoints/${jobId}`, {
      method: 'DELETE',
    });
  }

  // ========================================
  // QUEUE ENDPOINTS
  // ========================================

  // GET /api/queue - Queue status and worker info
  async getQueueStatus() {
    return this.request('/api/queue');
  }

  // POST /api/queue/spawn - Trigger remote worker spawn (debug)
  async spawnWorker() {
    return this.request('/api/queue/spawn', {
      method: 'POST',
    });
  }

  // POST /api/queue/cleanup - Trigger stale job cleanup (admin)
  async cleanupQueue(adminToken, dryRun = false) {
    const params = dryRun ? '?dry_run=true' : '';
    return this.request(`/api/queue/cleanup${params}`, {
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
  async pollJobStatus(jobId, onUpdate = null, maxAttempts = 120) {
    let attempts = 0;
    
    const poll = async () => {
      try {
        const status = await this.getJobStatus(jobId);
        
        if (onUpdate) {
          onUpdate(status);
        }
        
        if (status.status === 'completed') {
          return status;
        } else if (status.status === 'failed') {
          throw new Error(status.error || 'Job failed');
        } else if (attempts >= maxAttempts) {
          throw new Error('Job polling timeout');
        } else {
          attempts++;
          await new Promise(resolve => setTimeout(resolve, 2000));
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
  async streamJobProgress(jobId, onProgress, onComplete, onError) {
    const poll = async () => {
      try {
        const status = await this.getJobStatus(jobId);
        
        if (onProgress) {
          onProgress(status);
        }
        
        if (status.status === 'completed') {
          if (onComplete) onComplete(status);
          return;
        } else if (status.status === 'failed') {
          if (onError) onError(status.error || 'Job failed');
          return;
        }
        
        setTimeout(poll, 1000); // Poll every second
      } catch (error) {
        if (onError) onError(error);
      }
    };
    
    poll();
  }

  // Batch operations
  async batchDeleteJobs(jobIds) {
    const promises = jobIds.map(id => this.deleteJob(id));
    return Promise.allSettled(promises);
  }

  async batchRetryJobs(jobIds) {
    const promises = jobIds.map(id => this.retryJob(id));
    return Promise.allSettled(promises);
  }
}

// Export singleton instance
export const pyreflectAPI = new PyReflectAPI();
export default pyreflectAPI;