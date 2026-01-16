'use client';

import { useState, useEffect } from 'react';
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
  };
  is_local?: boolean;
  hf_url?: string;
}

interface InProgressStatus {
  name?: string | null;
  epochProgress?: { current: number; total: number } | null;
}

interface ExploreSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  userId?: string;
  onLoadSave: (params: { layers: FilmLayer[]; generator: GeneratorParams; training: TrainingParams }, result: GenerateResponse) => void;
  onRequestDownload: (saveId: string) => void;
  inProgress?: InProgressStatus | null;
}

export default function ExploreSidebar({
  isOpen,
  onClose,
  userId,
  onLoadSave,
  onRequestDownload,
  inProgress,
}: ExploreSidebarProps) {
  const [history, setHistory] = useState<SavedGeneration[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const progressPercent =
    inProgress?.epochProgress && inProgress.epochProgress.total > 0
      ? Math.min(100, Math.max(0, (inProgress.epochProgress.current / inProgress.epochProgress.total) * 100))
      : null;

  const requestDelete = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    setDeleteId(id);
  };

  const confirmDelete = async () => {
    if (!deleteId) return;
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/history/${deleteId}`, {
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

  useEffect(() => {
    if (!isOpen || !userId) return;

    const fetchHistory = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/history`,
          {
            headers: {
              'X-User-ID': userId,
            },
          }
        );

        if (!res.ok) throw new Error('Failed to fetch history');

        const data = await res.json();
        setHistory(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, [isOpen, userId]);

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

        <div className={styles.header}>
          <div className={styles.headerTop}>
            <span className={styles.title}>History</span>
            <button className={styles.closeBtn} onClick={onClose}>Close</button>
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
          {!userId ? (
            <div className={styles.emptyState}>Please sign in to view history</div>
          ) : loading ? (
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
                       <span className={styles.itemName} title={item.name}>
                         {item.name}
                         {item.is_local && <span style={{ fontSize: '10px', color: '#f59e0b', marginLeft: '6px' }}>(Local)</span>}
                       </span>
                    ) : (
                      <span className={styles.itemDate}>
                        {new Date(item.created_at).toLocaleString()}
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
                {item.name && <div className={styles.itemSubDate}>{new Date(item.created_at).toLocaleString()}</div>}
                
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
