'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { zip, strToU8 } from 'fflate';

import type { BundleSelection } from '@/components/DownloadBundleModal';
import type { ExportPngs, FilmLayer, GenerateResponse, GeneratorParams, TrainingParams } from '@/types';
import { base64ToUint8, estimateBase64Bytes } from '@/lib/base64';
import { captureChartPngs, EXPORT_CHARTS, type ExpandedCaptureMode } from '@/lib/chartsExport';

export type BundleParams = {
  layers: FilmLayer[];
  generator: GeneratorParams;
  training: TrainingParams;
};

export type BundlePayload = {
  params: BundleParams;
  result: GenerateResponse;
};

type BundleEstimate = {
  jsonBytes: number;
  pngBytes: number;
  pngNormalBytes: number;
  pngExpandedBytes: number;
  modelBytes: number | null;
  modelSource?: string | null;
  nrDataBytes: number | null;
  sldDataBytes: number | null;
  totalBytes: number | null;
  estimating: boolean;
};

type UseDownloadBundleArgs = {
  apiUrl: string;
  addLog: (message: string) => void;
  currentParams: BundleParams;
  graphData: GenerateResponse | null;
  userId?: string;
  onLoadSave: (params: BundleParams, result: GenerateResponse) => void;
};

export function useDownloadBundle({
  apiUrl,
  addLog,
  currentParams,
  graphData,
  userId,
  onLoadSave,
}: UseDownloadBundleArgs) {
  const [showBundleConfirm, setShowBundleConfirm] = useState(false);
  const [bundlePayload, setBundlePayload] = useState<BundlePayload | null>(null);
  const [isDownloadingBundle, setIsDownloadingBundle] = useState(false);
  const [bundleSelection, setBundleSelection] = useState<BundleSelection>({
    includeJson: true,
    includePngNormal: true,
    includePngExpanded: true,
    includeModel: true,
    includeNrData: true,
    includeSldData: true,
  });
  const [exportPngs, setExportPngs] = useState<ExportPngs | null>(null);
  const [shouldCapturePngs, setShouldCapturePngs] = useState(false);
  const [bundleEstimate, setBundleEstimate] = useState<BundleEstimate>({
    jsonBytes: 0,
    pngBytes: 0,
    pngNormalBytes: 0,
    pngExpandedBytes: 0,
    modelBytes: null,
    modelSource: null,
    nrDataBytes: null,
    sldDataBytes: null,
    totalBytes: null,
    estimating: false,
  });
  const [bundleEstimateError, setBundleEstimateError] = useState<string | null>(null);

  const bundlePngCacheRef = useRef<{
    payload: BundlePayload | null;
    pngs: ExportPngs;
    expandedMode: ExpandedCaptureMode;
    includeNormal: boolean;
    includeExpanded: boolean;
  } | null>(null);

  const resetPngs = useCallback(() => {
    setExportPngs(null);
    setShouldCapturePngs(false);
    bundlePngCacheRef.current = null;
  }, []);

  const requestAutoCapture = useCallback(() => {
    setShouldCapturePngs(true);
  }, []);

  const openBundleConfirm = useCallback((payload: BundlePayload) => {
    setBundlePayload(payload);
    setShowBundleConfirm(true);
    setBundleSelection({
      includeJson: true,
      includePngNormal: true,
      includePngExpanded: true,
      includeModel: Boolean(payload.result.model_id),
      includeNrData: Boolean(payload.result.model_id),
      includeSldData: Boolean(payload.result.model_id),
    });
    bundlePngCacheRef.current = null;
  }, []);

  const closeBundleConfirm = useCallback(() => {
    setShowBundleConfirm(false);
    setBundlePayload(null);
  }, []);

  useEffect(() => {
    if (!shouldCapturePngs || !graphData) return;
    let cancelled = false;

    const runCapture = async () => {
      try {
        const pngPayload = await captureChartPngs();
        if (!cancelled) {
          setExportPngs(pngPayload);
        }
      } catch {
        addLog('Warning: Failed to auto-capture PNGs.');
      } finally {
        if (!cancelled) {
          setShouldCapturePngs(false);
        }
      }
    };

    runCapture();
    return () => {
      cancelled = true;
    };
  }, [shouldCapturePngs, graphData, addLog]);

  useEffect(() => {
    if (!showBundleConfirm || !bundlePayload) return;
    let cancelled = false;

    const estimate = async () => {
      setBundleEstimateError(null);
      setBundleEstimate((prev) => ({ ...prev, estimating: true, totalBytes: null, modelBytes: null }));

      let pngBytes = 0;
      let pngNormalBytes = 0;
      let pngExpandedBytes = 0;
      let jsonBytes = 0;
      const embedPngsInJson =
        bundleSelection.includeJson &&
        !bundleSelection.includePngNormal &&
        !bundleSelection.includePngExpanded;
      const includeNormal = bundleSelection.includePngNormal || embedPngsInJson;
      const includeExpanded = bundleSelection.includePngExpanded || embedPngsInJson;
      const expandedMode: ExpandedCaptureMode = 'dpi';
      const needsPngs = includeNormal || includeExpanded;

      try {
        if (needsPngs) {
          const cached = bundlePngCacheRef.current;
          const pngPayload =
            cached &&
            cached.payload === bundlePayload &&
            cached.expandedMode === expandedMode &&
            cached.includeNormal === includeNormal &&
            cached.includeExpanded === includeExpanded
              ? cached.pngs
              : exportPngs ??
                await captureChartPngs({
                  expandedMode,
                  includeNormal,
                  includeExpanded,
                });

          if (bundleSelection.includeJson) {
            const { export_pngs: _exportPngs, ...resultWithoutPngs } = bundlePayload.result;
            void _exportPngs;
            const resultForJson =
              embedPngsInJson ? { ...bundlePayload.result, export_pngs: pngPayload } : resultWithoutPngs;
            const exportData = {
              params: {
                layers: bundlePayload.params.layers,
                generator: bundlePayload.params.generator,
                training: bundlePayload.params.training,
              },
              result: resultForJson,
            };
            const jsonStr = JSON.stringify(exportData);
            jsonBytes = new TextEncoder().encode(jsonStr).length;
          }

          const normalValues = bundleSelection.includePngNormal ? Object.values(pngPayload.normal) : [];
          const expandedValues = bundleSelection.includePngExpanded ? Object.values(pngPayload.expanded) : [];
          pngNormalBytes = normalValues.reduce((sum, base64) => sum + estimateBase64Bytes(base64), 0);
          pngExpandedBytes = expandedValues.reduce((sum, base64) => sum + estimateBase64Bytes(base64), 0);
          pngBytes = pngNormalBytes + pngExpandedBytes;
          bundlePngCacheRef.current = { payload: bundlePayload, pngs: pngPayload, expandedMode, includeNormal, includeExpanded };
        }
      } catch {
        setBundleEstimateError('Could not estimate PNG size.');
      }

      let modelBytes: number | null = null;
      let modelSource: string | null = null;
      let nrDataBytes: number | null = null;
      let sldDataBytes: number | null = null;
      
      if (bundlePayload.result.model_id) {
        // Fetch model info
        if (bundleSelection.includeModel) {
          try {
            const res = await fetch(`${apiUrl}/api/models/${bundlePayload.result.model_id}/info`, {
              headers: userId ? { 'X-User-ID': userId } : undefined,
            });
            if (res.ok) {
              const data = await res.json();
              if (typeof data.size_mb === 'number') {
                modelBytes = data.size_mb * 1024 * 1024;
              }
              modelSource = data.source || null;
            }
          } catch {
            modelSource = null;
          }
        }
        
        // Training data sizes are unknown (stored on HuggingFace)
        // Show as "Unknown" in the UI
        if (bundleSelection.includeNrData) {
          nrDataBytes = null;
        }
        if (bundleSelection.includeSldData) {
          sldDataBytes = null;
        }
      }

      const totalBytes = jsonBytes + pngBytes + (modelBytes ?? 0) + (nrDataBytes ?? 0) + (sldDataBytes ?? 0);

      if (cancelled) return;
      setBundleEstimate({
        jsonBytes,
        pngBytes,
        pngNormalBytes,
        pngExpandedBytes,
        modelBytes,
        modelSource,
        nrDataBytes,
        sldDataBytes,
        totalBytes,
        estimating: false,
      });
    };

    estimate();

    return () => {
      cancelled = true;
    };
  }, [showBundleConfirm, bundlePayload, exportPngs, bundleSelection, apiUrl, userId]);

  const handleExportAll = useCallback(async () => {
    if (!graphData) return;

    let pngPayload = exportPngs;
    if (!pngPayload) {
      try {
        pngPayload = await captureChartPngs();
        setExportPngs(pngPayload);
      } catch {
        addLog('Warning: PNG export capture failed.');
      }
    }

    const exportData = {
      params: currentParams,
      result: {
        ...graphData,
        export_pngs: pngPayload ?? graphData.export_pngs,
      },
    };

    const name = `pyreflect_export_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    const jsonStr = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    addLog('Exported full session data (params + results)');
  }, [graphData, exportPngs, currentParams, addLog]);

  const handleDownloadBundle = useCallback(
    async (payload?: BundlePayload, onCaptureComplete?: () => void) => {
      const resolvedResult = payload?.result ?? graphData;
      if (!resolvedResult) {
        addLog('Nothing to download yet.');
        return;
      }

      const resolvedParams = payload?.params ?? currentParams;

      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const baseName = resolvedResult.name
        ? resolvedResult.name.replace(/[^a-z0-9_-]+/gi, '_').slice(0, 32)
        : `pyreflect_bundle_${timestamp}`;
      const hasSelection =
        bundleSelection.includeJson ||
        bundleSelection.includePngNormal ||
        bundleSelection.includePngExpanded ||
        (bundleSelection.includeModel && Boolean(resolvedResult.model_id));

      if (!hasSelection) {
        addLog('No download items selected.');
        return;
      }

      const files: Record<string, Uint8Array> = {};

      try {
        addLog('Preparing download bundle...');
        await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
        const embedPngsInJson =
          bundleSelection.includeJson &&
          !bundleSelection.includePngNormal &&
          !bundleSelection.includePngExpanded;
        const includeNormal = bundleSelection.includePngNormal || embedPngsInJson;
        const includeExpanded = bundleSelection.includePngExpanded || embedPngsInJson;
        const expandedMode: ExpandedCaptureMode = bundleSelection.includePngExpanded ? 'fullscreen' : 'dpi';
        const needsPngs = includeNormal || includeExpanded;

        let pngPayload: ExportPngs | null = null;
        if (needsPngs) {
          const cached = bundlePngCacheRef.current;
          if (
            cached &&
            cached.payload === payload &&
            cached.expandedMode === expandedMode &&
            cached.includeNormal === includeNormal &&
            cached.includeExpanded === includeExpanded
          ) {
            pngPayload = cached.pngs;
          } else {
            const canReuseAutoCapture = expandedMode === 'dpi' && exportPngs !== null;
            pngPayload = canReuseAutoCapture
              ? exportPngs
              : await captureChartPngs({
                  expandedMode,
                  includeNormal,
                  includeExpanded,
                });
            bundlePngCacheRef.current = {
              payload: payload ?? null,
              pngs: pngPayload,
              expandedMode,
              includeNormal,
              includeExpanded,
            };
          }
        }

        // Signal that capture phase is done - overlay can dismiss now
        onCaptureComplete?.();

        if (bundleSelection.includeJson) {
          const { export_pngs: _exportPngs, ...resultWithoutPngs } = resolvedResult;
          void _exportPngs;
          const resultForJson =
            embedPngsInJson && pngPayload ? { ...resolvedResult, export_pngs: pngPayload } : resultWithoutPngs;
          const exportData = {
            params: resolvedParams,
            result: resultForJson,
          };
          files['output.json'] = strToU8(JSON.stringify(exportData));
        }

        if (pngPayload) {
          for (const chart of EXPORT_CHARTS) {
            if (bundleSelection.includePngNormal) {
              const normalBase64 = pngPayload.normal[chart.id];
              if (normalBase64) {
                files[chart.filename] = base64ToUint8(normalBase64);
              }
            }
            if (bundleSelection.includePngExpanded) {
              const expandedBase64 = pngPayload.expanded[chart.id];
              if (expandedBase64) {
                const expandedName = chart.filename.replace(/\.png$/i, '_expanded.png');
                files[expandedName] = base64ToUint8(expandedBase64);
              }
            }
            await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
          }
        }

        if (bundleSelection.includeModel && resolvedResult.model_id) {
          if (!userId) {
            addLog('Sign in to download model file.');
          } else {
            const modelRes = await fetch(`${apiUrl}/api/models/${resolvedResult.model_id}`, {
              headers: { 'X-User-ID': userId },
            });
          if (modelRes.ok) {
            const modelBuffer = await modelRes.arrayBuffer();
            files[`model_${resolvedResult.model_id}.pth`] = new Uint8Array(modelBuffer);
          } else {
            addLog('Model file not found for this run.');
          }
          }
        } else if (bundleSelection.includeModel && !resolvedResult.model_id) {
          addLog('No model file associated with this run.');
        }

        // Download NR training data
        if (bundleSelection.includeNrData && resolvedResult.model_id) {
          if (!userId) {
            addLog('Sign in to download NR training data.');
          } else {
            try {
              const nrRes = await fetch(`${apiUrl}/api/models/${resolvedResult.model_id}/training-data/nr_train`, {
                headers: { 'X-User-ID': userId },
              });
              if (nrRes.ok) {
                const nrBuffer = await nrRes.arrayBuffer();
                files[`nr_train_${resolvedResult.model_id}.npy`] = new Uint8Array(nrBuffer);
              } else {
                addLog('NR training data not found.');
              }
            } catch {
              addLog('Failed to fetch NR training data.');
            }
          }
        }

        // Download SLD training data
        if (bundleSelection.includeSldData && resolvedResult.model_id) {
          if (!userId) {
            addLog('Sign in to download SLD training data.');
          } else {
            try {
              const sldRes = await fetch(`${apiUrl}/api/models/${resolvedResult.model_id}/training-data/sld_train`, {
                headers: { 'X-User-ID': userId },
              });
              if (sldRes.ok) {
                const sldBuffer = await sldRes.arrayBuffer();
                files[`sld_train_${resolvedResult.model_id}.npy`] = new Uint8Array(sldBuffer);
              } else {
                addLog('SLD training data not found.');
              }
            } catch {
              addLog('Failed to fetch SLD training data.');
            }
          }
        }

        const zipData = await new Promise<Uint8Array>((resolve, reject) => {
          zip(files, { level: 6 }, (err, data) => {
            if (err) reject(err);
            else resolve(data);
          });
        });
        const blob = new Blob([zipData.buffer.slice(zipData.byteOffset, zipData.byteOffset + zipData.byteLength) as ArrayBuffer], { type: 'application/zip' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${baseName}.zip`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        addLog('Download bundle ready.');
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'Failed to build download bundle';
        addLog(`Download error: ${errorMsg}`);
      }
    },
    [graphData, currentParams, exportPngs, bundleSelection, addLog, apiUrl, userId]
  );

  const handleDownloadBundleClick = useCallback(() => {
    if (!graphData) {
      addLog('No results to download yet.');
      return;
    }
    openBundleConfirm({ params: currentParams, result: graphData });
  }, [graphData, currentParams, openBundleConfirm, addLog]);

  const handleSidebarDownloadRequest = useCallback(
    async (saveId: string) => {
      if (!userId) {
        alert('Please sign in to download history items.');
        return;
      }

      try {
        resetPngs();
        const res = await fetch(`${apiUrl}/api/history/${saveId}`, {
          headers: {
            'X-User-ID': userId,
          },
        });

        if (!res.ok) {
          throw new Error('Failed to load history item for download.');
        }

        const data = await res.json();
        if (!data?.params || !data?.result) {
          throw new Error('History item is missing data.');
        }

        onLoadSave(data.params, data.result);
        await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
        await new Promise((resolve) => setTimeout(resolve, 120));
        openBundleConfirm({ params: data.params, result: data.result });
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'Download failed';
        addLog(`Download error: ${errorMsg}`);
      }
    },
    [userId, apiUrl, onLoadSave, openBundleConfirm, addLog, resetPngs]
  );

  const handleConfirmBundleDownload = useCallback(async () => {
    if (!bundlePayload || isDownloadingBundle) return;
    setIsDownloadingBundle(true);
    closeBundleConfirm();
    await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
    try {
      await handleDownloadBundle(bundlePayload, () => {
        // Dismiss overlay once capture is done (bundling continues in background)
        setIsDownloadingBundle(false);
      });
    } catch {
      // Error already logged in handleDownloadBundle
    } finally {
      // Ensure overlay is dismissed even if callback wasn't reached
      setIsDownloadingBundle(false);
    }
  }, [bundlePayload, handleDownloadBundle, closeBundleConfirm, isDownloadingBundle]);

  return {
    showBundleConfirm,
    bundlePayload,
    isDownloadingBundle,
    bundleSelection,
    setBundleSelection,
    bundleEstimate,
    bundleEstimateError,
    closeBundleConfirm,
    openBundleConfirm,
    handleDownloadBundleClick,
    handleSidebarDownloadRequest,
    handleConfirmBundleDownload,
    handleExportAll,
    exportPngs,
    setExportPngs,
    requestAutoCapture,
    resetPngs,
  };
}
