'use client';

import { formatBytes } from '@/lib/format';
import InfoTooltip from '@/components/InfoTooltip';

export type BundleSelection = {
  includeJson: boolean;
  includePngNormal: boolean;
  includePngExpanded: boolean;
  includeModel: boolean;
  includeNrData: boolean;
  includeSldData: boolean;
};

type BundleEstimate = {
  jsonBytes: number;
  pngNormalBytes: number;
  pngExpandedBytes: number;
  modelBytes: number | null;
  modelSource?: string | null;
  nrDataBytes: number | null;
  sldDataBytes: number | null;
  totalBytes: number | null;
  estimating: boolean;
};

type DownloadBundleModalProps = {
  isOpen: boolean;
  hasPayload: boolean;
  hasModel: boolean;
  isDownloading?: boolean;
  selection: BundleSelection;
  estimate: BundleEstimate;
  estimateError: string | null;
  onClose: () => void;
  onConfirm: () => void;
  onSelectionChange: (patch: Partial<BundleSelection>) => void;
  progress?: number;
};

export default function DownloadBundleModal({
  isOpen,
  hasPayload,
  hasModel,
  isDownloading = false,
  selection,
  estimate,
  estimateError,
  onClose,
  onConfirm,
  onSelectionChange,
  progress = 0,
}: DownloadBundleModalProps) {
  if (!isOpen || !hasPayload) return null;

  const canDownload =
    selection.includeJson ||
    selection.includePngNormal ||
    selection.includePngExpanded ||
    (selection.includeModel && hasModel) ||
    (selection.includeNrData && hasModel) ||
    (selection.includeSldData && hasModel);

  return (
    <>
      <div className="model-download-overlay" onClick={onClose} />
      <div className="model-download-popup">
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', marginBottom: '16px', lineHeight: '1.4' }}>
          <div style={{ fontWeight: 600, marginBottom: '8px' }}>Download Bundle?</div>
          <div style={{ color: 'var(--text-muted)', fontSize: '11px', marginBottom: '10px' }}>
            Includes model (.pth), normal + expanded chart PNGs, and output.json. Model is pulled from Hugging Face if not local.
          </div>
          <div className="download-options">
            <label className="download-option">
              <span className="download-option__label">
                <input
                  type="checkbox"
                  className="download-checkbox"
                  checked={selection.includeJson}
                  onChange={(e) => onSelectionChange({ includeJson: e.target.checked })}
                />
                <span className="download-option__text">
                  output.json
                  <InfoTooltip hint="Includes params + results. If you deselect PNGs, they are embedded in output.json as base64." />
                </span>
              </span>
              <span className="download-option__size">
                {selection.includeJson
                  ? estimate.estimating
                    ? 'Estimating...'
                    : formatBytes(estimate.jsonBytes)
                  : 'Not selected'}
              </span>
            </label>
            <label className="download-option">
              <span className="download-option__label">
                <input
                  type="checkbox"
                  className="download-checkbox"
                  checked={selection.includePngNormal}
                  onChange={(e) => onSelectionChange({ includePngNormal: e.target.checked })}
                />
                <span className="download-option__text">
                  PNGs (normal)
                  <InfoTooltip hint="Exports per-chart PNG files at 1× scale." />
                </span>
              </span>
              <span className="download-option__size">
                {selection.includePngNormal
                  ? estimate.estimating
                    ? 'Estimating...'
                    : formatBytes(estimate.pngNormalBytes)
                  : 'Not selected'}
              </span>
            </label>
            <label className="download-option">
              <span className="download-option__label">
                <input
                  type="checkbox"
                  className="download-checkbox"
                  checked={selection.includePngExpanded}
                  onChange={(e) => onSelectionChange({ includePngExpanded: e.target.checked })}
                />
                <span className="download-option__text">
                  PNGs (expanded)
                  <InfoTooltip hint="Exports per-chart PNG files from the fullscreen expanded view." />
                </span>
              </span>
              <span className="download-option__size">
                {selection.includePngExpanded
                  ? estimate.estimating
                    ? 'Estimating...'
                    : formatBytes(estimate.pngExpandedBytes)
                  : 'Not selected'}
              </span>
            </label>
            <label className="download-option">
              <span className="download-option__label">
                <input
                  type="checkbox"
                  className="download-checkbox"
                  checked={selection.includeModel}
                  disabled={!hasModel}
                  onChange={(e) => onSelectionChange({ includeModel: e.target.checked })}
                />
                <span className="download-option__text">
                  Model (.pth)
                  <InfoTooltip hint="Downloads the model weights. If missing locally, it is pulled from Hugging Face if configured." />
                </span>
              </span>
              <span className="download-option__size">
                {hasModel
                  ? selection.includeModel
                    ? estimate.modelBytes !== null
                      ? formatBytes(estimate.modelBytes)
                      : estimate.estimating
                        ? 'Estimating...'
                        : 'Unknown'
                    : 'Not selected'
                  : 'Not available'}
              </span>
            </label>
            <label className="download-option">
              <span className="download-option__label">
                <input
                  type="checkbox"
                  className="download-checkbox"
                  checked={selection.includeNrData}
                  disabled={!hasModel}
                  onChange={(e) => onSelectionChange({ includeNrData: e.target.checked })}
                />
                <span className="download-option__text">
                  NR Data (.npy)
                  <InfoTooltip hint="Neutron reflectivity training curves used to train this model." />
                </span>
              </span>
              <span className="download-option__size">
                {hasModel
                  ? selection.includeNrData
                    ? estimate.nrDataBytes !== null
                      ? formatBytes(estimate.nrDataBytes)
                      : estimate.estimating
                        ? 'Estimating...'
                        : 'Unknown'
                    : 'Not selected'
                  : 'Not available'}
              </span>
            </label>
            <label className="download-option">
              <span className="download-option__label">
                <input
                  type="checkbox"
                  className="download-checkbox"
                  checked={selection.includeSldData}
                  disabled={!hasModel}
                  onChange={(e) => onSelectionChange({ includeSldData: e.target.checked })}
                />
                <span className="download-option__text">
                  SLD Data (.npy)
                  <InfoTooltip hint="Scattering length density profiles used to train this model." />
                </span>
              </span>
              <span className="download-option__size">
                {hasModel
                  ? selection.includeSldData
                    ? estimate.sldDataBytes !== null
                      ? formatBytes(estimate.sldDataBytes)
                      : estimate.estimating
                        ? 'Estimating...'
                        : 'Unknown'
                    : 'Not selected'
                  : 'Not available'}
              </span>
            </label>
          </div>
          <div style={{ fontSize: '12px' }}>
            Total:{' '}
            <span style={{ color: 'var(--text-secondary)' }}>
              {estimate.estimating ? 'Estimating...' : formatBytes(estimate.totalBytes)}
              {selection.includeModel && hasModel && estimate.modelBytes === null ? ' + model' : ''}
            </span>
          </div>
          {estimateError && (
            <div style={{ color: 'var(--text-muted)', fontSize: '11px', marginTop: '8px' }}>{estimateError}</div>
          )}
        </div>
        <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
          <button className="btn btn--outline" onClick={onClose} style={{ padding: '6px 12px', fontSize: '11px' }}>
            CANCEL
          </button>
          <button
            className="btn"
            onClick={onConfirm}
            style={{ 
              padding: '6px 12px', 
              fontSize: '11px',
              background: isDownloading 
                ? `linear-gradient(to right, var(--text-link) ${progress * 100}%, var(--surface-hover) ${progress * 100}%)` 
                : undefined,
              borderColor: isDownloading ? 'transparent' : undefined,
              color: isDownloading ? '#fff' : undefined,
              transition: 'background 0.2s ease',
            }}
            disabled={!canDownload || isDownloading}
          >
            {isDownloading ? `DOWNLOADING ${(progress * 100).toFixed(0)}%` : 'DOWNLOAD'}
          </button>
        </div>
      </div>
    </>
  );
}
