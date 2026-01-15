'use client';

import { useMemo, useCallback } from 'react';
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
import { GenerateResponse } from '@/types';
import styles from './GraphDisplay.module.css';

interface GraphDisplayProps {
  data: GenerateResponse | null;
}

// Export utilities
function downloadCSV(filename: string, headers: string[], rows: (number | string)[][]) {
  const csv = [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

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

  // Export handlers
  const exportNRData = useCallback(() => {
    if (!data?.nr) return;
    const gt = data.nr.groundTruth ?? (data.nr as unknown as { reflectivity: number[] }).reflectivity;
    const comp = data.nr.computed ?? gt;
    if (!gt) return;
    downloadCSV('nr_data.csv', ['Q', 'GroundTruth', 'ComputedNR'], 
      data.nr.q.map((q, i) => [q, gt[i], comp[i]]));
  }, [data]);

  const exportSLDData = useCallback(() => {
    if (!data?.sld) return;
    const gt = data.sld.groundTruth ?? (data.sld as unknown as { sld: number[] }).sld;
    const pred = data.sld.predicted ?? gt;
    if (!gt) return;
    downloadCSV('sld_profile.csv', ['z', 'GroundTruth', 'Predicted'],
      data.sld.z.map((z, i) => [z, gt[i], pred[i]]));
  }, [data]);

  const exportTrainingData = useCallback(() => {
    if (!data?.training) return;
    downloadCSV('training_loss.csv', ['Epoch', 'TrainingLoss', 'ValidationLoss'],
      data.training.epochs.map((e, i) => [e, data.training.trainingLoss[i], data.training.validationLoss[i]]));
  }, [data]);

  const exportChiData = useCallback(() => {
    if (!data?.chi) return;
    downloadCSV('chi_parameters.csv', ['Index', 'Predicted', 'Actual'],
      data.chi.map(c => [c.x, c.predicted, c.actual]));
  }, [data]);

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

      {/* Graphs Grid */}
      <div className="graph-container">
        {/* NR Curve */}
        <div className="graph-card">
          <div className="graph-card__header">
            <span className="graph-card__title">Neutron Reflectivity</span>
            <button className={styles.downloadBtn} onClick={exportNRData} title="Download CSV">↓</button>
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
                />
                <Line 
                  type="monotone" 
                  dataKey="computed"
                  name="Computed NR"
                  stroke="var(--color-gray-500, #888)" 
                  strokeWidth={1.5}
                  strokeDasharray="5 5"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* SLD Profile */}
        <div className="graph-card">
          <div className="graph-card__header">
            <span className="graph-card__title">SLD Profile</span>
            <button className={styles.downloadBtn} onClick={exportSLDData} title="Download CSV">↓</button>
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
                />
                <Line 
                  type="monotone" 
                  dataKey="predicted"
                  name="Predicted"
                  stroke="#cc4444"
                  strokeWidth={1.5}
                  strokeDasharray="5 5"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Training Loss */}
        <div className="graph-card">
          <div className="graph-card__header">
            <span className="graph-card__title">Training Loss</span>
            <button className={styles.downloadBtn} onClick={exportTrainingData} title="Download CSV">↓</button>
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
                />
                <Line 
                  type="monotone" 
                  dataKey="validation" 
                  name="Validation"
                  stroke="var(--color-gray-400)" 
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chi Parameters */}
        <div className="graph-card">
          <div className="graph-card__header">
            <span className="graph-card__title">Chi Parameters</span>
            <button className={styles.downloadBtn} onClick={exportChiData} title="Download CSV">↓</button>
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
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
