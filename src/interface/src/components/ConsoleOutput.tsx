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
  const [elapsedMs, setElapsedMs] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isGenerating || startTimeMs === null) {
      return;
    }
    const update = () => setElapsedMs(Date.now() - startTimeMs);
    update();
    const interval = setInterval(update, 500);
    return () => clearInterval(interval);
  }, [isGenerating, startTimeMs]);

  useEffect(() => {
    if (scrollRef.current && isExpanded) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, isExpanded]);

  const formatElapsed = (ms: number) => {
    const totalSeconds = Math.max(0, Math.floor(ms / 1000));
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };
  const displayElapsed = isGenerating && startTimeMs !== null ? elapsedMs : 0;

  return (
    <div className={styles.container}>
      <button 
        className={styles.header} 
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span className={styles.title}>
          <span className={styles.icon}>{isExpanded ? '▼' : '▶'}</span>
          Console
        </span>
        <span className={styles.meta}>
          <span className={styles.elapsed}>
            {isGenerating && startTimeMs !== null ? formatElapsed(displayElapsed) : '--:--:--'}
          </span>
          <span className={styles.count}>{logs.length}</span>
        </span>
      </button>
      
      {isExpanded && (
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
