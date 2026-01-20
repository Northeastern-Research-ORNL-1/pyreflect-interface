'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import styles from './ExploreSidebar.module.css';
import { GenerateResponse, FilmLayer, GeneratorParams, TrainingParams } from '@/types';

interface SavedGeneration {
  _id: string;
  user_id: string;
  name?: string;
  created_at: string;
  params: {
    layers: FilmLayer[];
    generator: GeneratorParams;
    training: TrainingParams;
  };
  result?: {
      metrics: {
        mse: number;
        r2: number;
        mae: number;
      };
      model_id?: string;
      model_size_mb?: number;
      timing?: {
        total?: number;
      };
      timings?: {
        total: number;
        training: number;
        inference: number;
      };
  };
  is_local?: boolean;
  hf_url?: string;
}

interface QueuedJob {
  job_id: string;
  status: string;
  enqueued_at?: string;
  started_at?: string;
  ended_at?: string;
  exc_info?: string | null;
  result?: {
    model_id?: string;
  };
  meta?: {
    status?: string;
    progress?: { epoch: number; total: number };
    logs?: string[];
    user_id?: string;
    name?: string;
    retried_from?: string;
    updated_at?: string;
    completed_at?: string;
    stop_requested?: boolean;
  };
}


// Helper to format duration
function formatSeconds(sec: number) {
  const safe = Number.isFinite(sec) ? sec : 0;
  const total = Math.max(0, Math.round(safe));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return `${h}h ${m}m ${s}s`;
}

function formatDuration(startStr?: string) {
  if (!startStr) return '';
  const start = new Date(startStr).getTime();
  const now = Date.now();
  const diff = Math.max(0, Math.floor((now - start) / 1000));
  
  const h = Math.floor(diff / 3600);
  const m = Math.floor((diff % 3600) / 60);
  const s = diff % 60;
  
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function formatDurationBetween(startStr?: string, endStr?: string) {
  if (!startStr) return '';
  const start = new Date(startStr).getTime();
  const end = endStr ? new Date(endStr).getTime() : Date.now();
  const diff = Math.max(0, Math.floor((end - start) / 1000));

  const h = Math.floor(diff / 3600);
  const m = Math.floor((diff % 3600) / 60);
  const s = diff % 60;

  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

interface QueueInfo {
  available: boolean;
  queued_jobs: number;
  job_ids: string[];
}

interface InProgressStatus {
  name?: string | null;
  epochProgress?: { current: number; total: number } | null;
}

interface ExploreSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  userId?: string;
  onResetLocal?: () => void;
  onLoadSave: (params: { layers: FilmLayer[]; generator: GeneratorParams; training: TrainingParams }, result: GenerateResponse) => void;
  onRequestDownload: (saveId: string) => void;
  inProgress?: InProgressStatus | null;
}

export default function ExploreSidebar({
  isOpen,
  onClose,
  userId,
  onResetLocal,
  onLoadSave,
  onRequestDownload,
  inProgress,
}: ExploreSidebarProps) {
  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  const [history, setHistory] = useState<SavedGeneration[]>([]);
  const [loading, setLoading] = useState(false);
  const [historyLoadedOnce, setHistoryLoadedOnce] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [cancelJobId, setCancelJobId] = useState<string | null>(null);
  const [stopJobId, setStopJobId] = useState<string | null>(null);
  const [retryJobId, setRetryJobId] = useState<string | null>(null);
  const [deleteJobId, setDeleteJobId] = useState<string | null>(null);
  const [purgeJobsOpen, setPurgeJobsOpen] = useState(false);
  const [queuedJobs, setQueuedJobs] = useState<QueuedJob[]>([]);
  const claimedJobSeenRef = useRef<Set<string>>(new Set());
  const claimRetryAtRef = useRef<Record<string, number>>({});
  const savedToHistorySeenRef = useRef<Set<string>>(new Set());
  const movingToHistorySeenRef = useRef<Set<string>>(new Set());
  const noticeTimerRef = useRef<number | null>(null);
  const [historyMoveNotice, setHistoryMoveNotice] = useState<string | null>(null);
  const [editingHistoryId, setEditingHistoryId] = useState<string | null>(null);
  const [editingHistoryName, setEditingHistoryName] = useState('');
  const [editingJobId, setEditingJobId] = useState<string | null>(null);
  const [editingJobName, setEditingJobName] = useState('');
  const progressPercent =
    inProgress?.epochProgress && inProgress.epochProgress.total > 0
      ? Math.min(100, Math.max(0, (inProgress.epochProgress.current / inProgress.epochProgress.total) * 100))
      : null;

  const requestDelete = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    setDeleteId(id);
  };

  const requestCancelJob = (e: React.MouseEvent, jobId: string) => {
    e.stopPropagation();
    setCancelJobId(jobId);
  };

  const requestStopJob = (e: React.MouseEvent, jobId: string) => {
    e.stopPropagation();
    setStopJobId(jobId);
  };

  const requestRetryJob = (e: React.MouseEvent, jobId: string) => {
    e.stopPropagation();
    setRetryJobId(jobId);
  };

  const requestDeleteJob = (e: React.MouseEvent, jobId: string) => {
    e.stopPropagation();
    setDeleteJobId(jobId);
  };

  const confirmDelete = async () => {
    if (!deleteId) return;
    try {
      const res = await fetch(`${API_URL}/api/history/${deleteId}`, {
        method: 'DELETE',
        headers: {
          'X-User-ID': userId!,
        },
      });

      if (!res.ok) throw new Error('Failed to delete save');

      setHistory((prev) => prev.filter((item) => item._id !== deleteId));
      setDeleteId(null);
    } catch (err) {
      console.error(err);
      alert('Failed to delete save');
    }
  };

  const confirmCancelJob = async () => {
    if (!cancelJobId) return;
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    try {
      const res = await fetch(`${API_URL}/api/jobs/${cancelJobId}`, { method: 'DELETE' });
      if (!res.ok) {
        const detail = await res.text().catch(() => '');
        throw new Error(detail || 'Failed to cancel job');
      }
      setQueuedJobs((prev) => prev.filter((j) => j.job_id !== cancelJobId));
      setCancelJobId(null);
    } catch (err) {
      console.error(err);
      alert('Failed to cancel job');
    }
  };

  const confirmStopJob = async () => {
    if (!stopJobId) return;
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    try {
      const res = await fetch(`${API_URL}/api/jobs/${stopJobId}/stop`, { method: 'POST' });
      if (!res.ok) {
        const detail = await res.text().catch(() => '');
        throw new Error(detail || 'Failed to stop job');
      }
      setStopJobId(null);
    } catch (err) {
      console.error(err);
      alert('Failed to stop job');
    }
  };

  const confirmRetryJob = async () => {
    if (!retryJobId) return;
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    try {
      const res = await fetch(`${API_URL}/api/jobs/${retryJobId}/retry`, { method: 'POST' });
      if (!res.ok) {
        const raw = await res.text().catch(() => '');
        try {
          const parsed = JSON.parse(raw) as { detail?: string };
          throw new Error(parsed?.detail || raw || 'Failed to retry job');
        } catch {
          throw new Error(raw || 'Failed to retry job');
        }
      }
      setRetryJobId(null);
    } catch (err) {
      console.error(err);
      alert(err instanceof Error ? err.message : 'Failed to retry job');
    }
  };

  const confirmDeleteJob = async () => {
    if (!deleteJobId) return;
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    try {
      const res = await fetch(`${API_URL}/api/jobs/${deleteJobId}/delete`, { method: 'DELETE' });
      if (!res.ok) {
        const raw = await res.text().catch(() => '');
        try {
          const parsed = JSON.parse(raw) as { detail?: string };
          throw new Error(parsed?.detail || raw || 'Failed to delete job');
        } catch {
          throw new Error(raw || 'Failed to delete job');
        }
      }
      setQueuedJobs((prev) => prev.filter((j) => j.job_id !== deleteJobId));
      setDeleteJobId(null);
    } catch (err) {
      console.error(err);
      alert(err instanceof Error ? err.message : 'Failed to delete job');
    }
  };

  const confirmPurgeJobs = async () => {
    if (!userId) return;
    try {
      const res = await fetch(`${API_URL}/api/jobs/purge`, {
        method: 'DELETE',
        headers: { 'X-User-ID': userId },
      });
      if (!res.ok) {
        const raw = await res.text().catch(() => '');
        try {
          const parsed = JSON.parse(raw) as { detail?: string };
          throw new Error(parsed?.detail || raw || 'Failed to clear job cache');
        } catch {
          throw new Error(raw || 'Failed to clear job cache');
        }
      }
      setPurgeJobsOpen(false);
      setQueuedJobs([]);
    } catch (err) {
      console.error(err);
      alert(err instanceof Error ? err.message : 'Failed to clear job cache');
    }
  };

  const startEditHistoryName = (e: React.MouseEvent, item: SavedGeneration) => {
    e.stopPropagation();
    if (!userId) return;
    setEditingHistoryId(item._id);
    setEditingHistoryName(item.name ?? '');
  };

  const saveHistoryName = async () => {
    if (!userId || !editingHistoryId) return;
    const name = editingHistoryName.trim() || null;
    try {
      const res = await fetch(`${API_URL}/api/history/${editingHistoryId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'X-User-ID': userId,
        },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) {
        const raw = await res.text().catch(() => '');
        try {
          const parsed = JSON.parse(raw) as { detail?: string };
          throw new Error(parsed?.detail || raw || 'Failed to rename history item');
        } catch {
          throw new Error(raw || 'Failed to rename history item');
        }
      }
      setHistory((prev) => prev.map((h) => (h._id === editingHistoryId ? { ...h, name: name ?? undefined } : h)));
      setEditingHistoryId(null);
    } catch (err) {
      console.error(err);
      alert(err instanceof Error ? err.message : 'Failed to rename history item');
    }
  };

  const cancelEditHistoryName = () => {
    setEditingHistoryId(null);
    setEditingHistoryName('');
  };

  const startEditJobName = (e: React.MouseEvent, job: QueuedJob) => {
    e.stopPropagation();
    if (!userId) return;
    setEditingJobId(job.job_id);
    setEditingJobName(job.meta?.name ?? '');
  };

  const saveJobName = async () => {
    if (!userId || !editingJobId) return;
    const name = editingJobName.trim() || null;
    try {
      const res = await fetch(`${API_URL}/api/jobs/${editingJobId}/name`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'X-User-ID': userId,
        },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) {
        const raw = await res.text().catch(() => '');
        try {
          const parsed = JSON.parse(raw) as { detail?: string };
          throw new Error(parsed?.detail || raw || 'Failed to rename job');
        } catch {
          throw new Error(raw || 'Failed to rename job');
        }
      }
      setQueuedJobs((prev) =>
        prev.map((j) =>
          j.job_id === editingJobId ? { ...j, meta: { ...(j.meta ?? {}), name: name ?? undefined } } : j
        )
      );
      setEditingJobId(null);
    } catch (err) {
      console.error(err);
      alert(err instanceof Error ? err.message : 'Failed to rename job');
    }
  };

  const cancelEditJobName = () => {
    setEditingJobId(null);
    setEditingJobName('');
  };

  const fetchHistory = useCallback(async (opts?: { silent?: boolean }): Promise<SavedGeneration[] | null> => {
    if (!isOpen || !userId) return null;
    if (!opts?.silent) {
      setLoading(true);
      setError(null);
    }
    try {
      const res = await fetch(`${API_URL}/api/history`, {
        cache: 'no-store',
        headers: {
          'X-User-ID': userId,
        },
      });

      if (!res.ok) throw new Error('Failed to fetch history');

      const data = await res.json();
      setHistory(data);
      return data as SavedGeneration[];
    } catch (err) {
      if (!opts?.silent) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      }
      return null;
    } finally {
      // Mark that we've attempted at least one history fetch so the sidebar
      // doesn't briefly show the empty state before the first request runs.
      setHistoryLoadedOnce(true);
      if (!opts?.silent) {
        setLoading(false);
      }
    }
  }, [API_URL, isOpen, userId]);

  const showNotice = useCallback((message: string) => {
    setHistoryMoveNotice(message);
    if (noticeTimerRef.current) {
      window.clearTimeout(noticeTimerRef.current);
    }
    noticeTimerRef.current = window.setTimeout(() => setHistoryMoveNotice(null), 5000);
  }, []);

  useEffect(() => {
    return () => {
      if (noticeTimerRef.current) {
        window.clearTimeout(noticeTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  // Poll queue for active jobs
  useEffect(() => {
    if (!isOpen) return;

    const fetchQueue = async () => {
      try {
        const headers: Record<string, string> = {};
        if (userId) {
          headers['X-User-ID'] = userId;
        }
        const res = await fetch(`${API_URL}/api/queue`, { cache: 'no-store', headers });
        if (!res.ok) return;
        const data: QueueInfo = await res.json();
        
        if (!data.available || data.job_ids.length === 0) {
          setQueuedJobs([]);
          return;
        }

        // Fetch status for each job
        const jobs = await Promise.all(
          data.job_ids.slice(0, 10).map(async (jobId) => {
            try {
              const jobRes = await fetch(`${API_URL}/api/jobs/${jobId}`, { cache: 'no-store' });
              if (!jobRes.ok) return null;
              return await jobRes.json() as QueuedJob;
            } catch {
              return null;
            }
          })
        );
        const hydrated = jobs.filter((j): j is QueuedJob => j !== null);

        // If a user logs in mid-run, claim jobs that aren't associated with any user yet,
        // so the worker can save results to history when complete.
        if (userId) {
          // Claim any job once; claim endpoint also persists finished jobs to history.
          const toClaim = hydrated.filter((job) => !claimedJobSeenRef.current.has(job.job_id));
          for (const job of toClaim) claimedJobSeenRef.current.add(job.job_id);
          if (toClaim.length > 0) {
            await Promise.allSettled(
              toClaim.map((job) =>
                fetch(`${API_URL}/api/jobs/${job.job_id}/claim`, {
                  method: 'POST',
                  cache: 'no-store',
                  headers: { 'X-User-ID': userId },
                })
              )
            );
          }
        }

        let latestHistory: SavedGeneration[] | null = null;
        if (userId && hydrated.some((job) => job.status === 'finished')) {
          // While there are finished jobs, refresh history each poll so the UI can
          // reliably hide jobs only after they appear in history.
          latestHistory = await fetchHistory({ silent: true });
        }

        const historyModelIds = new Set(
          (latestHistory ?? history)
            .map((h) => h.result?.model_id)
            .filter((id): id is string => typeof id === 'string' && id.length > 0)
        );

        if (userId) {
          for (const job of hydrated) {
            if (job.status !== 'finished') continue;
            const modelId = job.result?.model_id;
            if (!modelId) continue;
            const label = job.meta?.name || modelId.slice(0, 8);
            if (!historyModelIds.has(modelId)) {
              // Retry claim for finished jobs until they appear in history (rate-limited).
              const last = claimRetryAtRef.current[job.job_id] ?? 0;
              const now = Date.now();
              if (now - last > 10_000) {
                claimRetryAtRef.current[job.job_id] = now;
                fetch(`${API_URL}/api/jobs/${job.job_id}/claim`, {
                  method: 'POST',
                  cache: 'no-store',
                  headers: { 'X-User-ID': userId },
                })
                  .then(() => fetchHistory({ silent: true }))
                  .catch(() => {});
              }
              if (movingToHistorySeenRef.current.has(modelId)) continue;
              movingToHistorySeenRef.current.add(modelId);
              showNotice(`"${label}" training complete - moving to history...`);
              continue;
            }
            if (savedToHistorySeenRef.current.has(modelId)) continue;
            savedToHistorySeenRef.current.add(modelId);
          }
        }

        setQueuedJobs(
          hydrated.filter((job) => {
            if (!userId) return true;
            if (job.status === 'queued' || job.status === 'started' || job.status === 'failed') return true;
            if (job.status !== 'finished') return true;
            const modelId = job.result?.model_id;
            if (!modelId) return true;
            return !historyModelIds.has(modelId);
          })
        );
      } catch {
        // Queue not available
      }
    };

    fetchQueue();
    const interval = setInterval(fetchQueue, 1000);
    return () => clearInterval(interval);
  }, [API_URL, fetchHistory, history, isOpen, userId]);

  const handleLoad = async (id: string) => {
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/history/${id}`, {
        headers: {
          'X-User-ID': userId!,
        },
      });
      
      if (!res.ok) throw new Error('Failed to load save');
      
      const data = await res.json();
      onLoadSave(data.params, data.result);
      onClose();
    } catch (err) {
      console.error(err);
      alert('Failed to load save');
    }
  };

  return (
    <>
      <div className={`${styles.backdrop} ${isOpen ? styles.open : ''}`} onClick={onClose} />
      <div className={`${styles.sidebar} ${isOpen ? styles.open : ''}`}>
        {deleteId && (
          <div className={styles.deletePopupOverlay}>
            <div className={styles.deletePopupContent}>
              <p className={styles.popupText}>
                Are you sure you want to delete this history item?
                {(() => {
                    const item = history.find(i => i._id === deleteId);
                    return item?.name ? <> <strong>&quot;{item.name}&quot;</strong></> : null;
                })()}
                <span style={{ display: 'block', marginTop: '6px', fontSize: '12px', color: 'var(--text-muted)' }}>
                  The associated model file will be deleted as well.
                </span>
              </p>
              <div className={styles.popupActions}>
                <button className={`${styles.popupBtn} ${styles.popupBtnCancel}`} onClick={() => setDeleteId(null)}>CANCEL</button>
                <button className={`${styles.popupBtn} ${styles.popupBtnDelete}`} onClick={confirmDelete}>DELETE</button>
              </div>
            </div>
          </div>
        )}

        {cancelJobId && (
          <div className={styles.deletePopupOverlay}>
            <div className={styles.deletePopupContent}>
              <p className={styles.popupText}>
                Cancel queued job?
                <strong>{cancelJobId.slice(0, 8)}</strong>
                <span style={{ display: 'block', marginTop: '6px', fontSize: '12px', color: 'var(--text-muted)' }}>
                  This only works before it starts running.
                </span>
              </p>
              <div className={styles.popupActions}>
                <button className={`${styles.popupBtn} ${styles.popupBtnCancel}`} onClick={() => setCancelJobId(null)}>KEEP</button>
                <button className={`${styles.popupBtn} ${styles.popupBtnDelete}`} onClick={confirmCancelJob}>CANCEL JOB</button>
              </div>
            </div>
          </div>
        )}

        {stopJobId && (
          <div className={styles.deletePopupOverlay}>
            <div className={styles.deletePopupContent}>
              <p className={styles.popupText}>
                Stop running job?
                <strong>{stopJobId.slice(0, 8)}</strong>
                <span style={{ display: 'block', marginTop: '6px', fontSize: '12px', color: 'var(--text-muted)' }}>
                  Training will stop after the current epoch completes.
                </span>
              </p>
              <div className={styles.popupActions}>
                <button className={`${styles.popupBtn} ${styles.popupBtnCancel}`} onClick={() => setStopJobId(null)}>KEEP RUNNING</button>
                <button className={`${styles.popupBtn} ${styles.popupBtnDelete}`} onClick={confirmStopJob}>STOP</button>
              </div>
            </div>
          </div>
        )}

        {retryJobId && (
          <div className={styles.deletePopupOverlay}>
            <div className={styles.deletePopupContent}>
              <p className={styles.popupText}>
                Retry job?
                <strong>{retryJobId.slice(0, 8)}</strong>
              </p>
              <div className={styles.popupActions}>
                <button className={`${styles.popupBtn} ${styles.popupBtnCancel}`} onClick={() => setRetryJobId(null)}>NO</button>
                <button className={`${styles.popupBtn} ${styles.popupBtnDelete}`} onClick={confirmRetryJob}>RETRY</button>
              </div>
            </div>
          </div>
        )}

        {deleteJobId && (
          <div className={styles.deletePopupOverlay}>
            <div className={styles.deletePopupContent}>
              <p className={styles.popupText}>
                Delete failed job?
                <strong>{deleteJobId.slice(0, 8)}</strong>
              </p>
              <div className={styles.popupActions}>
                <button className={`${styles.popupBtn} ${styles.popupBtnCancel}`} onClick={() => setDeleteJobId(null)}>KEEP</button>
                <button className={`${styles.popupBtn} ${styles.popupBtnDelete}`} onClick={confirmDeleteJob}>DELETE JOB</button>
              </div>
            </div>
          </div>
        )}

        {purgeJobsOpen && (
          <div className={styles.deletePopupOverlay}>
            <div className={styles.deletePopupContent}>
              <p className={styles.popupText}>
                Clear job cache?
                <span style={{ display: 'block', marginTop: '6px', fontSize: '12px', color: 'var(--text-muted)' }}>
                  Removes your finished/failed job records from Redis (does not delete history).
                </span>
              </p>
              <div className={styles.popupActions}>
                <button className={`${styles.popupBtn} ${styles.popupBtnCancel}`} onClick={() => setPurgeJobsOpen(false)}>CANCEL</button>
                <button className={`${styles.popupBtn} ${styles.popupBtnDelete}`} onClick={confirmPurgeJobs}>CLEAR</button>
              </div>
            </div>
          </div>
        )}

        <div className={styles.header}>
          <div className={styles.headerTop}>
            <span className={styles.title}>History</span>
            <button className={styles.closeBtn} onClick={onClose} aria-label="Close">
              ×
            </button>
          </div>
          <input 
            type="text" 
            placeholder="Search history..." 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className={styles.searchInput}
          />
        </div>

        <div className={styles.content}>
          {historyMoveNotice && (
            <div className={styles.notice}>{historyMoveNotice}</div>
          )}
          {inProgress && (
            <div className={`${styles.historyItem} ${styles.historyItemInProgress}`}>
              <div className={styles.itemHeader}>
                <span className={styles.itemName}>In progress</span>
                <span className={styles.progressStatus}>Generating</span>
              </div>
              {inProgress.name && (
                <div className={styles.itemSubDate}>{inProgress.name}</div>
              )}
              <div className={styles.paramsInfo}>
                {inProgress.epochProgress
                  ? `Epoch ${inProgress.epochProgress.current}/${inProgress.epochProgress.total}`
                  : 'Working...'}
              </div>
              <div className={styles.progressBar}>
                <div
                  className={`${styles.progressFill} ${progressPercent === null ? styles.progressIndeterminate : ''}`}
                  style={progressPercent === null ? undefined : { width: `${progressPercent}%` }}
                />
              </div>
            </div>
          )}
          
          {queuedJobs.length > 0 && (
            <div className={styles.jobsSection}>
                <div className={styles.sectionTitleRow}>
                  <div className={styles.sectionTitle}>Queue ({queuedJobs.length})</div>
                  {userId && (
                    <button
                      className={styles.jobActionBtn}
                      onClick={() => setPurgeJobsOpen(true)}
                      title="Clear job cache"
                    >
                      Clear cache
                    </button>
                  )}
                </div>
              {queuedJobs.map((job) => {
                const jobProgress = job.meta?.progress;
                const percent = jobProgress && jobProgress.total > 0
                  ? Math.min(100, Math.max(0, (jobProgress.epoch / jobProgress.total) * 100))
                  : null;
                const statusColor = job.status === 'failed' ? '#ef4444' : 
                                  job.status === 'finished' ? '#10b981' : 
                                  job.status === 'started' ? '#3b82f6' : '#f59e0b';
                const elapsed = job.status === 'queued'
                  ? formatDuration(job.enqueued_at)
                  : job.status === 'started'
                    ? formatDuration(job.started_at)
                    : formatDurationBetween(job.started_at, job.ended_at);
                const lastLog = job.meta?.logs && job.meta.logs.length > 0 ? job.meta.logs[job.meta.logs.length - 1] : null;
                const updatedAgo = job.meta?.updated_at ? formatDuration(job.meta.updated_at) : '';
                const failNote =
                  job.status === 'failed' && job.exc_info
                    ? job.exc_info.split('\n')[0]
                    : null;

                return (
                  <div key={job.job_id} className={`${styles.historyItem} ${styles.historyItemInProgress}`}>
                    <div className={styles.itemHeader}>
                      <div className={styles.jobTitleRow}>
                        <div className={styles.jobTitleBlock}>
                          {editingJobId === job.job_id ? (
                            <input
                              className={styles.inlineNameInput}
                              value={editingJobName}
                              autoFocus
                              onChange={(e) => setEditingJobName(e.target.value)}
                              onBlur={saveJobName}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                  e.preventDefault();
                                  void saveJobName();
                                } else if (e.key === 'Escape') {
                                  e.preventDefault();
                                  cancelEditJobName();
                                }
                              }}
                              placeholder="Untitled"
                            />
                          ) : (
                            <span
                              className={styles.jobTitle}
                              onDoubleClick={(e) => startEditJobName(e, job)}
                              title={userId ? 'Double-click to rename' : undefined}
                            >
                              {job.meta?.name || 'Untitled'}
                            </span>
                          )}
                          <span className={styles.jobId}>Job: {job.job_id.slice(0, 8)}</span>
                        </div>
                      </div>
                      <span className={styles.progressStatus} style={{ color: statusColor }}>
                        {job.status === 'queued' ? `Queued (${elapsed})`
                          : job.status === 'started' ? `Running (${elapsed})`
                          : elapsed ? `${job.status} (${elapsed})` : job.status}
                      </span>
                    </div>
                    {job.meta?.status === 'saving' && (
                      <div className={styles.itemSubDate}>Saving & Uploading...</div>
                    )}
                    <div className={styles.paramsInfo}>
                      {jobProgress
                        ? `Epoch ${jobProgress.epoch}/${jobProgress.total}`
                        : job.status === 'queued' 
                          ? 'Waiting...' 
                          : job.meta?.status || 'Processing...'}
                      {failNote && (
                        <div style={{ marginTop: '4px', fontSize: '11px', color: '#ef4444' }}>
                          {failNote}
                        </div>
                      )}
                      {lastLog && (
                        <div style={{ marginTop: '4px', fontSize: '11px', color: 'var(--text-muted)' }}>
                          {updatedAgo ? `[${updatedAgo}] ` : ''}{lastLog}
                        </div>
                      )}
                    </div>
                    {(job.status === 'started' || job.status === 'queued') && (
                      <div className={styles.progressBar}>
                        <div
                          className={`${styles.progressFill} ${percent === null ? styles.progressIndeterminate : ''}`}
                          style={{ 
                            width: percent === null ? undefined : `${percent}%`,
                            backgroundColor: statusColor
                          }}
                        />
                      </div>
                    )}
                    
                    <div className={styles.jobActionsFooter}>
                      {job.status === 'queued' && (
                        <button
                          className={`${styles.jobActionBtn} ${styles.jobActionDanger}`}
                          onClick={(e) => requestCancelJob(e, job.job_id)}
                          title="Cancel job"
                        >
                          Cancel
                        </button>
                      )}
                      {job.status === 'started' && !job.meta?.stop_requested && (
                        <button
                          className={`${styles.jobActionBtn} ${styles.jobActionDanger}`}
                          onClick={(e) => requestStopJob(e, job.job_id)}
                          title="Stop training after current epoch"
                        >
                          Stop
                        </button>
                      )}
                      {job.status === 'started' && job.meta?.stop_requested && (
                        <span className={styles.stoppingLabel}>Stopping...</span>
                      )}
                      {job.status === 'failed' && (
                        <>
                          <button
                            className={styles.jobActionBtn}
                            onClick={(e) => requestRetryJob(e, job.job_id)}
                            title="Retry job"
                          >
                            Retry
                          </button>
                          <button
                            className={`${styles.jobActionBtn} ${styles.jobActionDanger}`}
                            onClick={(e) => requestDeleteJob(e, job.job_id)}
                            title="Delete job"
                          >
                            Delete
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
              <div className={styles.divider} />
            </div>
          )}

          {!userId ? (
            <>
              {onResetLocal && (
                <div className={styles.jobsSection}>
                  <div className={styles.sectionTitleRow}>
                    <div className={styles.sectionTitle}></div>
                    <button
                      className={styles.jobActionBtn}
                      onClick={() => {
                        onResetLocal();
                        onClose();
                      }}
                      title="Clears local saved state (cache)"
                    >
                      Clear local cache
                    </button>
                  </div>
                </div>
              )}
              <div className={styles.emptyState}>
                History not saved, please log in
              </div>
            </>
          ) : !historyLoadedOnce ? (
            <div className={styles.emptyState}>Loading...</div>
          ) : error ? (
            <div className={styles.emptyState}>Error: {error}</div>
          ) : history.length === 0 ? (
            <div className={styles.emptyState}>No saved generations yet</div>
          ) : (
            history
              .filter(item => !searchTerm || (item.name && item.name.toLowerCase().includes(searchTerm.toLowerCase())))
              .map((item) => (
              <div 
                key={item._id} 
                className={styles.historyItem}
                onClick={() => handleLoad(item._id)}
              >
                <div className={styles.itemHeader}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1, minWidth: 0 }}>
	                    {item.name ? (
	                       editingHistoryId === item._id ? (
	                         <input
	                           className={styles.inlineNameInput}
	                           value={editingHistoryName}
	                           autoFocus
	                           onChange={(e) => setEditingHistoryName(e.target.value)}
	                           onBlur={saveHistoryName}
	                           onKeyDown={(e) => {
	                             if (e.key === 'Enter') {
	                               e.preventDefault();
	                               void saveHistoryName();
	                             } else if (e.key === 'Escape') {
	                               e.preventDefault();
	                               cancelEditHistoryName();
	                             }
	                           }}
	                           placeholder="Untitled"
	                         />
	                       ) : (
	                         <span
	                           className={styles.itemName}
	                           title={userId ? 'Double-click to rename' : item.name}
	                           onDoubleClick={(e) => startEditHistoryName(e, item)}
	                         >
	                           {item.name}
	                           {item.is_local && <span style={{ fontSize: '10px', color: '#f59e0b', marginLeft: '6px' }}>(Local)</span>}
	                         </span>
	                       )
	                    ) : (
	                      <span className={styles.itemDate} onDoubleClick={(e) => startEditHistoryName(e, item)} title={userId ? 'Double-click to rename' : undefined}>
	                        {new Date(item.created_at).toLocaleString()}
	                        {(() => {
	                          const elapsedSec = item.result?.timing?.total ?? item.result?.timings?.total;
	                          return typeof elapsedSec === 'number' ? ` • ${formatSeconds(elapsedSec)}` : '';
	                        })()}
	                        {item.is_local && <span style={{ fontSize: '10px', color: '#f59e0b', marginLeft: '6px' }}>(Local)</span>}
	                      </span>
	                    )}
                    {item.hf_url && (
                        <a 
                          href={item.hf_url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className={styles.deleteBtn}
                          title="View on Hugging Face"
                          style={{ opacity: 1, fontSize: '14px', marginLeft: 0, display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}
                          onClick={(e) => e.stopPropagation()}
                        >
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                            <polyline points="15 3 21 3 21 9"></polyline>
                            <line x1="10" y1="14" x2="21" y2="3"></line>
                          </svg>
                        </a>
                    )}
                    {item.result && (
                      <button 
                        className={styles.deleteBtn} 
                        onClick={(e) => {
                            e.stopPropagation();
                            onRequestDownload(item._id);
                        }}
                        title="Download Bundle"
                        style={{ opacity: 1, fontSize: '14px', marginLeft: 0 }}
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                          <polyline points="7 10 12 15 17 10"></polyline>
                          <line x1="12" y1="15" x2="12" y2="3"></line>
                        </svg>
                      </button>
                    )}
                  </div>

                  <button 
                    className={styles.deleteBtn} 
                    onClick={(e) => requestDelete(e, item._id)}
                    title="Delete"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="3 6 5 6 21 6"></polyline>
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                  </button>
                </div>
	                {item.name && <div className={styles.itemSubDate}>
	                  {new Date(item.created_at).toLocaleString()}
	                  {(() => {
	                    const elapsedSec = item.result?.timing?.total ?? item.result?.timings?.total;
	                    return typeof elapsedSec === 'number' ? ` • ${formatSeconds(elapsedSec)}` : '';
	                  })()}
	                </div>}
	                
	                <div className={styles.paramsInfo}>
	                  {item.params.layers.length} Layers • {item.params.generator.numCurves} Curves
	                </div>

                {item.result?.metrics && (
                  <div className={styles.itemMetrics}>
                    <div className={styles.metric}>
                      <span className={styles.metricLabel}>MSE</span>
                      <span className={styles.metricValue}>{item.result.metrics.mse.toFixed(4)}</span>
                    </div>
                    <div className={styles.metric}>
                      <span className={styles.metricLabel}>R²</span>
                      <span className={styles.metricValue}>{item.result.metrics.r2.toFixed(3)}</span>
                    </div>
                     <div className={styles.metric}>
                      <span className={styles.metricLabel}>MAE</span>
                      <span className={styles.metricValue}>{item.result.metrics.mae.toFixed(4)}</span>
                    </div>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </>
  );
}
