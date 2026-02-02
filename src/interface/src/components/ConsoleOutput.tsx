'use client';

import { useState, useRef, useEffect } from 'react';
import styles from './ConsoleOutput.module.css';

interface ConsoleOutputProps {
  logs: string[];
  isGenerating: boolean;
  startTimeMs: number | null;
}

export default function ConsoleOutput({ logs, isGenerating, startTimeMs }: ConsoleOutputProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (startTimeMs === null) return;
    const update = () => setElapsedMs(Date.now() - startTimeMs);
    update();
    const interval = setInterval(update, 500);
    return () => clearInterval(interval);
  }, [startTimeMs]);

  useEffect(() => {
    if (scrollRef.current && (isExpanded || isFullscreen)) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, isExpanded, isFullscreen]);

  const formatElapsed = (ms: number) => {
    const totalSeconds = Math.max(0, Math.floor(ms / 1000));
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };
  const displayElapsed = startTimeMs !== null ? elapsedMs : 0;

  const toggleFullscreen = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsFullscreen(!isFullscreen);
    // Ensure expanded if going fullscreen
    if (!isFullscreen) setIsExpanded(true);
  };

  return (
    <div className={`${styles.container} ${isFullscreen ? styles.fullscreen : ''}`}>
      <div 
        className={styles.header} 
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className={styles.headerLeft}>
          <span className={styles.title}>
            <span className={styles.icon}>{isExpanded || isFullscreen ? '▼' : '▶'}</span>
            Console Logs
          </span>
          <span className={styles.meta}>
            <span className={styles.elapsed}>
              {startTimeMs !== null ? formatElapsed(displayElapsed) : '--:--:--'}
            </span>
            <span className={styles.count}>{logs.length}</span>
          </span>
        </div>
        
        <button 
          className={styles.fullscreenBtn}
          onClick={toggleFullscreen}
          title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
        >
          {isFullscreen ? '✕' : '⛶'}
        </button>
      </div>
      
      {(isExpanded || isFullscreen) && (
        <div className={styles.content} ref={scrollRef}>
          {logs.length === 0 ? (
            <div className={styles.empty}>No logs yet</div>
          ) : (
            logs.map((log, i) => (
              <div key={i} className={styles.logLine}>
                {log}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
