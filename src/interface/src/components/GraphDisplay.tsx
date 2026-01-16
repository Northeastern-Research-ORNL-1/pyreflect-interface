'use client';

import { useMemo, useCallback, useState, useEffect, useRef } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  Legend,
} from 'recharts';
import { toPng } from 'html-to-image';
import { GenerateResponse } from '@/types';
import styles from './GraphDisplay.module.css';

interface GraphDisplayProps {
  data: GenerateResponse | null;
}

type SetExpandedCardEventDetail = {
  cardId: string | null;
  exporting?: boolean;
};

export default function GraphDisplay({ data }: GraphDisplayProps) {
  const nrChartData = useMemo(() => {
    if (!data?.nr) return [];
    // Backward compatibility: handle old format with 'reflectivity' field
    const gt = data.nr.groundTruth ?? (data.nr as unknown as { reflectivity: number[] }).reflectivity;
    const comp = data.nr.computed ?? gt;
    if (!gt) return [];
    return data.nr.q.map((q, i) => ({
      q,
      groundTruth: Math.log10(Math.max(gt[i], 1e-10)),
      computed: Math.log10(Math.max(comp[i], 1e-10)),
    }));
  }, [data]);

  const sldChartData = useMemo(() => {
    if (!data?.sld) return [];
    // Backward compatibility: handle old format with 'sld' field
    const gt = data.sld.groundTruth ?? (data.sld as unknown as { sld: number[] }).sld;
    const pred = data.sld.predicted ?? gt;
    if (!gt) return [];
    return data.sld.z.map((z, i) => ({
      z,
      groundTruth: gt[i],
      predicted: pred[i],
    }));
  }, [data]);

  const trainingChartData = useMemo(() => {
    if (!data?.training) return [];
    return data.training.epochs.map((epoch, i) => ({
      epoch,
      training: data.training.trainingLoss[i],
      validation: data.training.validationLoss[i],
    }));
  }, [data]);

  const chiChartData = useMemo(() => {
    if (!data?.chi) return [];
    return data.chi;
  }, [data]);

  // Refs for graph cards to support image export
  const nrCardRef = useRef<HTMLDivElement>(null);
  const sldCardRef = useRef<HTMLDivElement>(null);
  const trainingCardRef = useRef<HTMLDivElement>(null);
  const chiCardRef = useRef<HTMLDivElement>(null);

  const handleDownloadImage = useCallback(async (ref: React.RefObject<HTMLDivElement | null>, filename: string) => {
    if (ref.current === null) return;

    try {
      // Small delay to ensure any rendering is complete
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const { clientWidth, clientHeight } = ref.current;
      
      const dataUrl = await toPng(ref.current, {
        cacheBust: true,
        backgroundColor: '#000000', // Ensure dark background
        width: clientWidth,
        height: clientHeight,
        style: {
          transform: 'none', // Avoid scaling issues
          // Reset positioning that might cause offsets in the capture
          position: 'static', 
          top: '0',
          left: '0',
          right: 'auto',
          bottom: 'auto',
          margin: '0',
          width: `${clientWidth}px`,
          height: `${clientHeight}px`,
          // Disable animation which might confuse the capture
          animation: 'none',
        }
      });
      
      const link = document.createElement('a');
      link.download = `${filename}.png`;
      link.href = dataUrl;
      link.click();
    } catch (err) {
      console.error('Failed to download chart image:', err);
    }
  }, []);

  // Fullscreen expansion state
  const [expandedCard, setExpandedCard] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const expandedCardRef = useRef<string | null>(null);
  const isExportingRef = useRef(false);
  const prevExpandedCardRef = useRef<string | null>(null);

  useEffect(() => {
    expandedCardRef.current = expandedCard;
  }, [expandedCard]);

  useEffect(() => {
    isExportingRef.current = isExporting;
  }, [isExporting]);

  useEffect(() => {
    const handler = (event: Event) => {
      const custom = event as CustomEvent<SetExpandedCardEventDetail>;
      const detail = custom.detail;
      if (!detail || typeof detail !== 'object' || !('cardId' in detail)) return;

      const nextExporting = Boolean(detail.exporting);
      if (nextExporting) {
        if (!isExportingRef.current) {
          prevExpandedCardRef.current = expandedCardRef.current;
        }
        setIsExporting(true);
        setExpandedCard(detail.cardId);
        return;
      }

      setIsExporting(false);
      const restoreCard = prevExpandedCardRef.current;
      prevExpandedCardRef.current = null;
      setExpandedCard(restoreCard ?? detail.cardId);
    };

    window.addEventListener('pyreflect:set-expanded-card', handler);
    return () => window.removeEventListener('pyreflect:set-expanded-card', handler);
  }, []);

  const toggleExpand = useCallback((cardId: string) => {
    setExpandedCard(prev => prev === cardId ? null : cardId);
  }, []);

  const getCardClassName = useCallback((cardId: string) => {
    if (expandedCard === cardId) {
      return `graph-card ${styles.expandedCard} ${isExporting ? styles.exportingCard : ''}`;
    }
    return 'graph-card';
  }, [expandedCard, isExporting]);

  // Handle Escape key to close expanded card
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && expandedCard) {
        setExpandedCard(null);
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [expandedCard]);

  if (!data) {
    return (
      <div className={styles.empty}>
        <div className={styles.emptyIcon}>◇</div>
        <p>Adjust parameters and click <strong>GENERATE</strong> to visualize results</p>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {/* Metrics Row */}
      <div className="metrics">
        <div className="metric">
          <div className="metric__label">MSE</div>
          <div className="metric__value">
            {data.metrics.mse.toFixed(4)}
          </div>
        </div>
        <div className="metric">
          <div className="metric__label">R²</div>
          <div className="metric__value">
            {data.metrics.r2.toFixed(4)}
          </div>
        </div>
        <div className="metric">
          <div className="metric__label">MAE</div>
          <div className="metric__value">
            {data.metrics.mae.toFixed(4)}
          </div>
        </div>
      </div>

      {/* Fullscreen Backdrop */}
      {expandedCard && !isExporting && (
        <div 
          className={styles.backdrop} 
          onClick={() => setExpandedCard(null)}
        />
      )}

      {/* Graphs Grid */}
      <div className={expandedCard ? `graph-container ${styles.expandedContainer}` : 'graph-container'}>
        {/* NR Curve */}
        <div className={getCardClassName('nr')} ref={nrCardRef} data-export-id="nr">
          <div className="graph-card__header">
            <span className="graph-card__title">Neutron Reflectivity</span>
            <div className={styles.headerActions}>
              <button 
                className={styles.downloadBtn} 
                onClick={() => handleDownloadImage(nrCardRef, 'neutron_reflectivity')} 
                title="Download PNG"
              >
                <span>↓</span><span className={styles.btnLabel}>PNG</span>
              </button>
              <button 
                className={styles.expandBtn} 
                onClick={() => toggleExpand('nr')} 
                title={expandedCard === 'nr' ? 'Collapse' : 'Expand'}
              >
                <span>{expandedCard === 'nr' ? '×' : '⛶'}</span>
                <span className={styles.btnLabel}>{expandedCard === 'nr' ? 'Close' : 'Expand'}</span>
              </button>
            </div>
          </div>
          <div className="graph-card__content">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={nrChartData} margin={{ top: 20, right: 20, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                <XAxis 
                  dataKey="q" 
                  scale="log"
                  domain={['auto', 'auto']}
                  tick={{ fontSize: 10, fontFamily: 'var(--font-mono)' }}
                  tickFormatter={(v) => v.toFixed(3)}
                  label={{ value: 'Q (Å⁻¹)', position: 'bottom', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                />
                <YAxis 
                  domain={['auto', 'auto']}
                  tick={{ fontSize: 10, fontFamily: 'var(--font-mono)' }}
                  tickFormatter={(v) => v.toFixed(1)}
                  label={{ value: 'log₁₀(R)', angle: -90, position: 'insideLeft', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    background: 'var(--bg-primary)', 
                    border: '1px solid var(--border-color)',
                    fontFamily: 'var(--font-mono)',
                    fontSize: 11,
                  }}
                  labelFormatter={(label) => `Q: ${Number(label).toFixed(4)} Å⁻¹`}
                />
                <Legend 
                  verticalAlign="top"
                  height={24}
                  wrapperStyle={{ fontSize: 10, fontFamily: 'var(--font-mono)' }}
                />
	              <Line 
	                  type="monotone" 
	                  dataKey="groundTruth"
	                  name="Ground Truth"
	                  stroke="var(--text-primary)" 
	                  strokeWidth={1.5}
	                  dot={false}
	                  isAnimationActive={!isExporting}
	                />
	                <Line 
	                  type="monotone" 
	                  dataKey="computed"
	                  name="Computed NR"
	                  stroke="var(--color-gray-500, #888)" 
	                  strokeWidth={1.5}
	                  strokeDasharray="5 5"
	                  dot={false}
	                  isAnimationActive={!isExporting}
	                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* SLD Profile */}
        <div className={getCardClassName('sld')} ref={sldCardRef} data-export-id="sld">
          <div className="graph-card__header">
            <span className="graph-card__title">SLD Profile</span>
            <div className={styles.headerActions}>
              <button 
                className={styles.downloadBtn} 
                onClick={() => handleDownloadImage(sldCardRef, 'sld_profile')} 
                title="Download PNG"
              >
                <span>↓</span><span className={styles.btnLabel}>PNG</span>
              </button>
              <button 
                className={styles.expandBtn} 
                onClick={() => toggleExpand('sld')} 
                title={expandedCard === 'sld' ? 'Collapse' : 'Expand'}
              >
                <span>{expandedCard === 'sld' ? '×' : '⛶'}</span>
                <span className={styles.btnLabel}>{expandedCard === 'sld' ? 'Close' : 'Expand'}</span>
              </button>
            </div>
          </div>
          <div className="graph-card__content">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sldChartData} margin={{ top: 20, right: 20, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                <XAxis 
                  dataKey="z" 
                  tick={{ fontSize: 10, fontFamily: 'var(--font-mono)' }}
                  tickFormatter={(v) => Math.round(v).toString()}
                  label={{ value: 'z (Å)', position: 'bottom', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                />
                <YAxis 
                  tick={{ fontSize: 10, fontFamily: 'var(--font-mono)' }}
                  tickFormatter={(v) => v.toFixed(1)}
                  label={{ value: 'SLD (10⁻⁶ Å⁻²)', angle: -90, position: 'insideLeft', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    background: 'var(--bg-primary)', 
                    border: '1px solid var(--border-color)',
                    fontFamily: 'var(--font-mono)',
                    fontSize: 11,
                  }}
                  labelFormatter={(label) => `z: ${Math.round(Number(label))} Å`}
                />
                <Legend 
                  verticalAlign="top"
                  height={24}
                  wrapperStyle={{ fontSize: 10, fontFamily: 'var(--font-mono)' }}
                />
	                <Line 
	                  type="monotone" 
	                  dataKey="groundTruth"
	                  name="GroundTruth"
	                  stroke="var(--text-primary)" 
	                  strokeWidth={1.5}
	                  dot={false}
	                  isAnimationActive={!isExporting}
	                />
	                <Line 
	                  type="monotone" 
	                  dataKey="predicted"
	                  name="Predicted"
	                  stroke="#cc4444"
	                  strokeWidth={1.5}
	                  strokeDasharray="5 5"
	                  dot={false}
	                  isAnimationActive={!isExporting}
	                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Training Loss */}
        <div className={getCardClassName('training')} ref={trainingCardRef} data-export-id="training">
          <div className="graph-card__header">
            <span className="graph-card__title">Training Loss</span>
            <div className={styles.headerActions}>
              <button 
                className={styles.downloadBtn} 
                onClick={() => handleDownloadImage(trainingCardRef, 'training_loss')} 
                title="Download PNG"
              >
                <span>↓</span><span className={styles.btnLabel}>PNG</span>
              </button>
              <button 
                className={styles.expandBtn} 
                onClick={() => toggleExpand('training')} 
                title={expandedCard === 'training' ? 'Collapse' : 'Expand'}
              >
                <span>{expandedCard === 'training' ? '×' : '⛶'}</span>
                <span className={styles.btnLabel}>{expandedCard === 'training' ? 'Close' : 'Expand'}</span>
              </button>
            </div>
          </div>
          <div className="graph-card__content">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trainingChartData} margin={{ top: 20, right: 20, left: 10, bottom: 30 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                <XAxis 
                  dataKey="epoch" 
                  tick={{ fontSize: 10, fontFamily: 'var(--font-mono)' }}
                  label={{ value: 'Epoch', position: 'bottom', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                />
                <YAxis 
                  tick={{ fontSize: 10, fontFamily: 'var(--font-mono)' }}
                  tickFormatter={(v) => v.toFixed(2)}
                  label={{ value: 'Loss', angle: -90, position: 'insideLeft', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    background: 'var(--bg-primary)', 
                    border: '1px solid var(--border-color)',
                    fontFamily: 'var(--font-mono)',
                    fontSize: 11,
                  }}
                />
                <Legend 
                  verticalAlign="top"
                  height={24}
                  wrapperStyle={{ fontSize: 10, fontFamily: 'var(--font-mono)' }}
                />
	                <Line 
	                  type="monotone" 
	                  dataKey="training" 
	                  name="Training"
	                  stroke="var(--text-primary)" 
	                  strokeWidth={1.5}
	                  dot={false}
	                  isAnimationActive={!isExporting}
	                />
	                <Line 
	                  type="monotone" 
	                  dataKey="validation" 
	                  name="Validation"
	                  stroke="var(--color-gray-400)" 
	                  strokeWidth={1.5}
	                  strokeDasharray="4 4"
	                  dot={false}
	                  isAnimationActive={!isExporting}
	                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chi Parameters */}
        <div className={getCardClassName('chi')} ref={chiCardRef} data-export-id="chi">
          <div className="graph-card__header">
            <span className="graph-card__title">Chi Parameters</span>
            <div className={styles.headerActions}>
              <button 
                className={styles.downloadBtn} 
                onClick={() => handleDownloadImage(chiCardRef, 'chi_parameters')} 
                title="Download PNG"
              >
                <span>↓</span><span className={styles.btnLabel}>PNG</span>
              </button>
              <button 
                className={styles.expandBtn} 
                onClick={() => toggleExpand('chi')} 
                title={expandedCard === 'chi' ? 'Collapse' : 'Expand'}
              >
                <span>{expandedCard === 'chi' ? '×' : '⛶'}</span>
                <span className={styles.btnLabel}>{expandedCard === 'chi' ? 'Close' : 'Expand'}</span>
              </button>
            </div>
          </div>
          <div className="graph-card__content">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                <XAxis 
                  dataKey="actual"
                  type="number"
                  domain={['auto', 'auto']}
                  tick={{ fontSize: 10, fontFamily: 'var(--font-mono)' }}
                  tickFormatter={(v) => v.toFixed(2)}
                  label={{ value: 'Actual χ', position: 'bottom', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                />
                <YAxis 
                  dataKey="predicted"
                  type="number"
                  domain={['auto', 'auto']}
                  tick={{ fontSize: 10, fontFamily: 'var(--font-mono)' }}
                  tickFormatter={(v) => v.toFixed(2)}
                  label={{ value: 'Predicted χ', angle: -90, position: 'insideLeft', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    background: '#000', 
                    border: '1px solid #fff',
                    fontFamily: 'var(--font-mono)',
                    fontSize: 11,
                    padding: '8px 12px',
                    color: '#fff',
                  }}
                  itemStyle={{ color: '#fff' }}
                  labelStyle={{ color: '#fff' }}
                  formatter={(value, name) => [value != null ? Number(value).toFixed(4) : '', name === 'actual' ? 'Actual χ' : 'Predicted χ']}
                />
	                <Scatter
	                  data={chiChartData}
	                  fill="var(--text-primary)"
	                  isAnimationActive={!isExporting}
	                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
