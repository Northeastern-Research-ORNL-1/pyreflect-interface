'use client';

import { useRef, useCallback, useState, useMemo } from 'react';
import styles from './RangeSlider.module.css';

interface RangeSliderProps {
  /** Current central value */
  value: number;
  /** Minimum bound (left handle) */
  boundMin: number | null;
  /** Maximum bound (right handle) */
  boundMax: number | null;
  /** Minimum possible value for slider */
  min: number;
  /** Maximum possible value for slider */
  max: number;
  /** Step increment */
  step: number;
  /** Callback when value changes */
  onValueChange: (value: number) => void;
  /** Callback when bounds change (null = no bound) */
  onBoundsChange: (boundMin: number | null, boundMax: number | null) => void;
  /** Whether the slider is disabled */
  disabled?: boolean;
  /** Number of decimal places for display */
  decimals?: number;
}

export default function RangeSlider({
  value,
  boundMin,
  boundMax,
  min,
  max,
  step,
  onValueChange,
  onBoundsChange,
  disabled = false,
  decimals = 2,
}: RangeSliderProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<'value' | 'left' | 'right' | null>(null);

  const range = max - min;
  const valuePercent = ((value - min) / range) * 100;
  
  // Use value as fallback when no bounds set
  const effectiveMin = boundMin ?? value;
  const effectiveMax = boundMax ?? value;
  
  const leftPercent = ((effectiveMin - min) / range) * 100;
  const rightPercent = ((effectiveMax - min) / range) * 100;

  const hasBounds = boundMin !== null || boundMax !== null;

  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
  
  const snap = useMemo(() => {
    return (v: number) => Math.round(v / step) * step;
  }, [step]);
  
  const format = (v: number) => v.toFixed(decimals);

  const getValueFromX = useCallback(
    (clientX: number): number => {
      if (!trackRef.current) return value;
      const rect = trackRef.current.getBoundingClientRect();
      const percent = clamp((clientX - rect.left) / rect.width, 0, 1);
      return snap(min + percent * range);
    },
    [min, range, snap, value]
  );

  const handlePointerDown = useCallback(
    (e: React.PointerEvent, handle: 'value' | 'left' | 'right') => {
      if (disabled) return;
      e.preventDefault();
      e.stopPropagation();
      setDragging(handle);
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    },
    [disabled]
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragging || disabled) return;
      const newPos = getValueFromX(e.clientX);

      if (dragging === 'value') {
        // Move center value, clamped within bounds if they exist
        const lo = boundMin ?? min;
        const hi = boundMax ?? max;
        const clampedValue = clamp(newPos, lo, hi);
        onValueChange(clampedValue);
      } else if (dragging === 'left') {
        // Adjust left bound, can't exceed value or go below min
        const newMin = clamp(newPos, min, value);
        onBoundsChange(snap(newMin), boundMax);
      } else if (dragging === 'right') {
        // Adjust right bound, can't go below value or exceed max
        const newMax = clamp(newPos, value, max);
        onBoundsChange(boundMin, snap(newMax));
      }
    },
    [dragging, disabled, getValueFromX, value, boundMin, boundMax, min, max, onValueChange, onBoundsChange, snap]
  );

  const handlePointerUp = useCallback(
    (e: React.PointerEvent) => {
      if (dragging) {
        (e.target as HTMLElement).releasePointerCapture(e.pointerId);
        setDragging(null);
      }
    },
    [dragging]
  );

  // Handle track click to move value
  const handleTrackClick = useCallback(
    (e: React.MouseEvent) => {
      if (disabled) return;
      if ((e.target as HTMLElement).closest(`.${styles.handle}`)) return;
      const newValue = getValueFromX(e.clientX);
      const lo = boundMin ?? min;
      const hi = boundMax ?? max;
      const clampedValue = clamp(newValue, lo, hi);
      onValueChange(clampedValue);
    },
    [disabled, getValueFromX, min, max, boundMin, boundMax, onValueChange]
  );

  const clearBounds = useCallback(() => {
    onBoundsChange(null, null);
  }, [onBoundsChange]);

  return (
    <div className={`${styles.container} ${disabled ? styles.disabled : ''}`}>
      <div
        ref={trackRef}
        className={styles.track}
        onClick={handleTrackClick}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
      >
        {/* Range highlight */}
        {hasBounds && (
          <div
            className={styles.range}
            style={{
              left: `${leftPercent}%`,
              width: `${rightPercent - leftPercent}%`,
            }}
          />
        )}

        {/* Left bound handle - always visible */}
        <div
          className={`${styles.handle} ${styles.handleBound} ${styles.handleLeft} ${dragging === 'left' ? styles.handleActive : ''} ${!hasBounds ? styles.handleInactive : ''}`}
          style={{ left: `${leftPercent}%` }}
          onPointerDown={(e) => handlePointerDown(e, 'left')}
          title={hasBounds ? `Min: ${format(effectiveMin)}` : 'Drag left to set min bound'}
        >
          <div className={styles.handleGrip} />
        </div>

        {/* Center value handle */}
        <div
          className={`${styles.handle} ${styles.handleValue} ${dragging === 'value' ? styles.handleActive : ''}`}
          style={{ left: `${valuePercent}%` }}
          onPointerDown={(e) => handlePointerDown(e, 'value')}
          title={`Value: ${format(value)}`}
        />

        {/* Right bound handle - always visible */}
        <div
          className={`${styles.handle} ${styles.handleBound} ${styles.handleRight} ${dragging === 'right' ? styles.handleActive : ''} ${!hasBounds ? styles.handleInactive : ''}`}
          style={{ left: `${rightPercent}%` }}
          onPointerDown={(e) => handlePointerDown(e, 'right')}
          title={hasBounds ? `Max: ${format(effectiveMax)}` : 'Drag right to set max bound'}
        >
          <div className={styles.handleGrip} />
        </div>
      </div>

      {/* Bounds indicator */}
      {hasBounds && (
        <div className={styles.boundsInfo}>
          <span className={styles.boundsLabel}>
            [{format(effectiveMin)}, {format(effectiveMax)}]
          </span>
          <button
            className={styles.clearButton}
            onClick={clearBounds}
            title="Clear bounds"
            type="button"
          >
            ×
          </button>
        </div>
      )}
    </div>
  );
}
