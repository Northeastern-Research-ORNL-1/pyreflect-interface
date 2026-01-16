'use client';

import { useState, useRef } from 'react';
import { createPortal } from 'react-dom';
import styles from './InfoTooltip.module.css';

interface InfoTooltipProps {
  hint: string;
}

export default function InfoTooltip({ hint }: InfoTooltipProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const iconRef = useRef<HTMLSpanElement>(null);

  const handleMouseEnter = () => {
    if (iconRef.current) {
      const rect = iconRef.current.getBoundingClientRect();
      setPosition({
        top: rect.bottom + 4,
        left: rect.left,
      });
    }
    setIsVisible(true);
  };

  const tooltip = isVisible
    ? createPortal(
        <div className={styles.tooltip} style={{ top: position.top, left: position.left }}>
          {hint}
        </div>,
        document.body
      )
    : null;

  return (
    <>
      <span 
        ref={iconRef}
        className={styles.container}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={() => setIsVisible(false)}
      >
        <span className={styles.icon}>ⓘ</span>
      </span>
      {tooltip}
    </>
  );
}
