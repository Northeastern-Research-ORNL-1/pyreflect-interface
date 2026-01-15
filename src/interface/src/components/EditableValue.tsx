'use client';

import { useState, useRef, useEffect, type KeyboardEvent } from 'react';
import styles from './EditableValue.module.css';

interface EditableValueProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  decimals?: number;
  disabled?: boolean;
}

export default function EditableValue({
  value,
  onChange,
  min,
  max,
  step = 1,
  decimals = 2,
  disabled = false,
}: EditableValueProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [inputValue, setInputValue] = useState(value.toFixed(decimals));
  const inputRef = useRef<HTMLInputElement>(null);
  const effectiveMin = 0;
  const effectiveMax = 999999;

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  useEffect(() => {
    if (!isEditing) {
      setInputValue(value.toFixed(decimals));
    }
  }, [value, decimals, isEditing]);

  const handleClick = () => {
    if (!disabled) {
      setIsEditing(true);
    }
  };

  const commitValue = () => {
    let newValue = parseFloat(inputValue);
    if (isNaN(newValue)) {
      newValue = value;
    }
    newValue = Math.max(effectiveMin, newValue);
    newValue = Math.min(effectiveMax, newValue);
    onChange(newValue);
    setIsEditing(false);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      commitValue();
    } else if (e.key === 'Escape') {
      setInputValue(value.toFixed(decimals));
      setIsEditing(false);
    }
  };

  if (isEditing) {
    return (
      <input
        ref={inputRef}
        type="number"
        className={styles.input}
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onBlur={commitValue}
        onKeyDown={handleKeyDown}
        min={effectiveMin}
        max={effectiveMax}
        step={step}
      />
    );
  }

  return (
    <span
      className={`${styles.value} ${disabled ? styles.disabled : ''}`}
      onClick={handleClick}
      title={disabled ? undefined : 'Click to edit'}
    >
      {value.toFixed(decimals)}
    </span>
  );
}
