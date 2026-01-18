'use client';

import { useEffect, useRef, useState } from 'react';
import Image from 'next/image';
import { signIn, signOut } from 'next-auth/react';
import type { Session } from 'next-auth';

type EpochProgress = { current: number; total: number } | null;

type WorkerInfo = {
  name: string;
  state: string;
};

type AppHeaderProps = {
  appVersion: string;
  isProduction: boolean;
  isGenerating: boolean;
  epochProgress: EpochProgress;
  hasGraphData: boolean;
  session: Session | null;
  onOpenHistory: () => void;
  onImportJson: () => void;
  onExportJson: () => void;
  onDownloadBundle: () => void;
};

export default function AppHeader({
  appVersion,
  isProduction,
  isGenerating,
  epochProgress,
  hasGraphData,
  session,
  onOpenHistory,
  onImportJson,
  onExportJson,
  onDownloadBundle,
}: AppHeaderProps) {
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showActionsMenu, setShowActionsMenu] = useState(false);
  const [showJsonMenu, setShowJsonMenu] = useState(false);
  const [showJsonMenuMobile, setShowJsonMenuMobile] = useState(false);
  const [workers, setWorkers] = useState<WorkerInfo[]>([]);
  const jsonMenuRef = useRef<HTMLDivElement>(null);

  // Poll for worker info
  useEffect(() => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const fetchWorkers = async () => {
      try {
        const res = await fetch(`${API_URL}/api/queue`, { cache: 'no-store' });
        if (res.ok) {
          const data = await res.json();
          setWorkers(data.workers || []);
        }
      } catch {
        // Queue not available
      }
    };
    fetchWorkers();
    const interval = setInterval(fetchWorkers, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (jsonMenuRef.current && !jsonMenuRef.current.contains(event.target as Node)) {
        setShowJsonMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Determine worker type: GPU (colab), CPU, or none
  const hasGpuWorker = workers.some(w => w.name.toLowerCase().includes('colab') || w.name.toLowerCase().includes('gpu'));
  const hasCpuWorker = workers.some(w => !w.name.toLowerCase().includes('colab') && !w.name.toLowerCase().includes('gpu'));
  const workerDotColor = hasGpuWorker ? '#10b981' : hasCpuWorker ? '#3b82f6' : '#6b7280';
  const workerTooltip = hasGpuWorker 
    ? 'GPU worker connected (Colab)' 
    : hasCpuWorker 
      ? 'CPU worker connected' 
      : 'No workers connected';

  return (
    <header className="header">
      <div className="header__logo">
        <span>◇</span>
        <span>PYREFLECT</span>
        <span className="header__version">{appVersion}</span>
        {isProduction && (
          <span className="header__version" style={{ color: '#f59e0b', marginLeft: '8px' }}>
            PROD
          </span>
        )}
        <span
          className={`status ${isGenerating ? 'status--training' : 'status--active'}`}
          style={{ marginLeft: '12px' }}
        >
          <span className="status__dot"></span>
          <span className="header__status-text">
            {isGenerating
              ? epochProgress
                ? `Training... (${epochProgress.current}/${epochProgress.total})`
                : 'Training...'
              : 'Ready'}
          </span>
          {!isGenerating && (
            <span
              className="worker-dot"
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: workerDotColor,
                marginLeft: '8px',
                display: 'inline-block',
              }}
              title={workerTooltip}
            />
          )}
        </span>
      </div>
      <nav className="header__nav">
        <div className="header__actions-desktop">
          <a
            className="header__export-btn"
            href="https://github.com/Northeastern-Research-ORNL-1/pyreflect-interface"
            target="_blank"
            rel="noopener noreferrer"
            title="View on GitHub"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M12 .5C5.65.5.5 5.82.5 12.38c0 5.25 3.44 9.7 8.21 11.27.6.11.82-.27.82-.6 0-.3-.01-1.1-.02-2.16-3.34.75-4.04-1.66-4.04-1.66-.55-1.44-1.34-1.82-1.34-1.82-1.09-.77.08-.76.08-.76 1.2.09 1.83 1.27 1.83 1.27 1.07 1.88 2.8 1.34 3.49 1.03.11-.8.42-1.34.76-1.65-2.66-.31-5.47-1.36-5.47-6.06 0-1.34.46-2.44 1.23-3.31-.12-.31-.53-1.57.12-3.27 0 0 1.01-.33 3.3 1.26a11.2 11.2 0 0 1 3-.41c1.02 0 2.04.14 3 .41 2.29-1.59 3.3-1.26 3.3-1.26.65 1.7.24 2.96.12 3.27.77.87 1.23 1.97 1.23 3.31 0 4.71-2.81 5.75-5.49 6.05.43.38.81 1.13.81 2.28 0 1.65-.01 2.98-.01 3.39 0 .33.22.72.83.6 4.76-1.57 8.2-6.02 8.2-11.27C23.5 5.82 18.35.5 12 .5z" />
            </svg>
            <span className="header__btn-label">GitHub</span>
          </a>
          <button className="header__export-btn" onClick={onOpenHistory}>
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
            </svg>
            <span className="header__btn-label">Explore</span>
          </button>
          <div className="header__menu" ref={jsonMenuRef}>
            <button className="header__export-btn" onClick={() => setShowJsonMenu((prev) => !prev)}>
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M1.5 12s3.6-7 10.5-7 10.5 7 10.5 7-3.6 7-10.5 7-10.5-7-10.5-7z" />
                <circle cx="12" cy="12" r="3" />
              </svg>
              <span className="header__btn-label">View</span>
            </button>
            {showJsonMenu && (
              <div className="header__dropdown">
                <button
                  className="header__dropdown-item"
                  onClick={() => {
                    onImportJson();
                    setShowJsonMenu(false);
                  }}
                >
                  <span>↑</span> Import JSON
                </button>
                {hasGraphData && (
                  <button
                    className="header__dropdown-item"
                    onClick={() => {
                      onExportJson();
                      setShowJsonMenu(false);
                    }}
                  >
                    <span>↓</span> Export JSON
                  </button>
                )}
              </div>
            )}
          </div>
          {hasGraphData && (
            <button className="header__export-btn" onClick={onDownloadBundle}>
              <span>↓</span>
              <span className="header__btn-label">Download</span>
            </button>
          )}
        </div>

        <div className="header__actions-mobile">
          <button
            className="header__export-btn"
            onClick={() => {
              setShowActionsMenu((prev) => !prev);
              if (showActionsMenu) {
                setShowJsonMenuMobile(false);
              }
            }}
          >
            <span>≡</span>
          </button>
          {showActionsMenu && (
            <div className="header__dropdown">
              <a
                className="header__dropdown-item"
                href="https://github.com/Northeastern-Research-ORNL-1/pyreflect-interface"
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => setShowActionsMenu(false)}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                  <path d="M12 .5C5.65.5.5 5.82.5 12.38c0 5.25 3.44 9.7 8.21 11.27.6.11.82-.27.82-.6 0-.3-.01-1.1-.02-2.16-3.34.75-4.04-1.66-4.04-1.66-.55-1.44-1.34-1.82-1.34-1.82-1.09-.77.08-.76.08-.76 1.2.09 1.83 1.27 1.83 1.27 1.07 1.88 2.8 1.34 3.49 1.03.11-.8.42-1.34.76-1.65-2.66-.31-5.47-1.36-5.47-6.06 0-1.34.46-2.44 1.23-3.31-.12-.31-.53-1.57.12-3.27 0 0 1.01-.33 3.3 1.26a11.2 11.2 0 0 1 3-.41c1.02 0 2.04.14 3 .41 2.29-1.59 3.3-1.26 3.3-1.26.65 1.7.24 2.96.12 3.27.77.87 1.23 1.97 1.23 3.31 0 4.71-2.81 5.75-5.49 6.05.43.38.81 1.13.81 2.28 0 1.65-.01 2.98-.01 3.39 0 .33.22.72.83.6 4.76-1.57 8.2-6.02 8.2-11.27C23.5 5.82 18.35.5 12 .5z" />
                </svg>
                GitHub
              </a>
              <button
                className="header__dropdown-item"
                onClick={() => {
                  onOpenHistory();
                  setShowActionsMenu(false);
                }}
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
                </svg>
                History
              </button>
              <button
                className="header__dropdown-item"
                onClick={() => setShowJsonMenuMobile((prev) => !prev)}
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M1.5 12s3.6-7 10.5-7 10.5 7 10.5 7-3.6 7-10.5 7-10.5-7-10.5-7z" />
                  <circle cx="12" cy="12" r="3" />
                </svg>
                <span>{showJsonMenuMobile ? '▾' : '▸'}</span> JSON
              </button>
              {showJsonMenuMobile && (
                <div className="header__dropdown-sub">
                  <button
                    className="header__dropdown-item header__dropdown-item--sub"
                    onClick={() => {
                      onImportJson();
                      setShowJsonMenuMobile(false);
                      setShowActionsMenu(false);
                    }}
                  >
                    <span>↑</span> Import
                  </button>
                  {hasGraphData && (
                    <button
                      className="header__dropdown-item header__dropdown-item--sub"
                      onClick={() => {
                        onExportJson();
                        setShowJsonMenuMobile(false);
                        setShowActionsMenu(false);
                      }}
                    >
                      <span>↓</span> Export
                    </button>
                  )}
                </div>
              )}
              {hasGraphData && (
                <button
                  className="header__dropdown-item"
                  onClick={() => {
                    onDownloadBundle();
                    setShowActionsMenu(false);
                    setShowJsonMenuMobile(false);
                  }}
                >
                  <span>↓</span> Download
                </button>
              )}
            </div>
          )}
        </div>

        {session ? (
          <div className="header__user">
            {session.user?.image && (
              <button
                type="button"
                className="header__avatar"
                onClick={() => setShowUserMenu((prev) => !prev)}
                aria-label="User menu"
              >
                <Image
                  src={session.user.image}
                  alt=""
                  width={32}
                  height={32}
                  style={{
                    borderRadius: '50%',
                    border: '1px solid var(--text-primary)',
                  }}
                />
              </button>
            )}
            {showUserMenu && (
              <button
                className="header__export-btn header__export-btn--danger"
                onClick={() => {
                  setShowUserMenu(false);
                  signOut();
                }}
              >
                <span>→</span>
                <span className="header__btn-label">Sign out</span>
              </button>
            )}
          </div>
        ) : (
          <button className="header__export-btn" onClick={() => signIn('github')}>
            <span>←</span>
            <span className="header__btn-label">Sign in</span>
          </button>
        )}
      </nav>
    </header>
  );
}
