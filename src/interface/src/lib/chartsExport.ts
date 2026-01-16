import { toPng } from 'html-to-image';
import type { ExportPngs } from '@/types';
import { dataUrlToBase64 } from './base64';

export const EXPORT_CHARTS = [
  { id: 'nr', filename: 'neutron_reflectivity.png' },
  { id: 'sld', filename: 'sld_profile.png' },
  { id: 'training', filename: 'training_loss.png' },
  { id: 'chi', filename: 'chi_parameters.png' },
];

export type ExpandedCaptureMode = 'dpi' | 'fullscreen';

type CaptureChartPngsOptions = {
  expandedMode?: ExpandedCaptureMode;
  includeNormal?: boolean;
  includeExpanded?: boolean;
};

async function waitForCharts(): Promise<void> {
  for (let i = 0; i < 8; i += 1) {
    const nodes = document.querySelectorAll('[data-export-id]');
    if (nodes.length > 0) return;
    await new Promise((resolve) => setTimeout(resolve, 80));
  }
}

function setExpandedCardForExport(cardId: string | null, exporting: boolean) {
  window.dispatchEvent(
    new CustomEvent('pyreflect:set-expanded-card', {
      detail: { cardId, exporting },
    })
  );
}

async function settleLayout() {
  await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
  await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
  await new Promise((resolve) => setTimeout(resolve, 50));
}

async function waitForExportChartReady(chartId: string, timeoutMs = 5000): Promise<HTMLElement | null> {
  const start = performance.now();
  while (performance.now() - start < timeoutMs) {
    const node = document.querySelector(`[data-export-id="${chartId}"]`) as HTMLElement | null;
    if (node) {
      const { clientWidth, clientHeight } = node;
      const wrapper = node.querySelector('.recharts-wrapper') as HTMLElement | null;
      const svg = node.querySelector('svg') as SVGElement | null;
      const hasWrapper = wrapper ? wrapper.clientWidth > 0 && wrapper.clientHeight > 0 : true;
      const hasSvg = svg ? svg.getBoundingClientRect().width > 0 && svg.getBoundingClientRect().height > 0 : false;
      if (clientWidth > 0 && clientHeight > 0 && hasWrapper && hasSvg) {
        await new Promise((resolve) => setTimeout(resolve, 150));
        return node;
      }
    }
    await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
  }
  return document.querySelector(`[data-export-id="${chartId}"]`) as HTMLElement | null;
}

export async function captureChartPngs(options: CaptureChartPngsOptions = {}): Promise<ExportPngs> {
  await waitForCharts();
  const normal: Record<string, string> = {};
  const expanded: Record<string, string> = {};
  const includeNormal = options.includeNormal ?? true;
  const includeExpanded = options.includeExpanded ?? true;
  const expandedMode: ExpandedCaptureMode = options.expandedMode ?? 'dpi';
  const useFullscreenExpanded = includeExpanded && expandedMode === 'fullscreen';

  if (useFullscreenExpanded) {
    setExpandedCardForExport(null, true);
    await settleLayout();
  }

  try {
    if (includeNormal) {
      for (const chart of EXPORT_CHARTS) {
        const node = document.querySelector(`[data-export-id="${chart.id}"]`) as HTMLElement | null;
        if (!node) continue;
        const { clientWidth, clientHeight } = node;
        if (clientWidth === 0 || clientHeight === 0) continue;

        const normalUrl = await toPng(node, {
          cacheBust: true,
          backgroundColor: '#000000',
          width: clientWidth,
          height: clientHeight,
          pixelRatio: 1,
          style: {
            transform: 'none',
            position: 'static',
            top: '0',
            left: '0',
            right: 'auto',
            bottom: 'auto',
            margin: '0',
            width: `${clientWidth}px`,
            height: `${clientHeight}px`,
            animation: 'none',
          },
        });

        normal[chart.id] = dataUrlToBase64(normalUrl);
      }
    }

    if (includeExpanded) {
      if (expandedMode === 'fullscreen') {
        for (const chart of EXPORT_CHARTS) {
          setExpandedCardForExport(chart.id, true);
          await settleLayout();

          const node = await waitForExportChartReady(chart.id);
          if (!node) continue;
          const { clientWidth, clientHeight } = node;
          if (clientWidth === 0 || clientHeight === 0) continue;

          const expandedUrl = await toPng(node, {
            cacheBust: true,
            backgroundColor: '#000000',
            width: clientWidth,
            height: clientHeight,
            pixelRatio: 1,
            style: {
              transform: 'none',
              position: 'static',
              top: '0',
              left: '0',
              right: 'auto',
              bottom: 'auto',
              margin: '0',
              width: `${clientWidth}px`,
              height: `${clientHeight}px`,
              animation: 'none',
            },
          });

          expanded[chart.id] = dataUrlToBase64(expandedUrl);
        }
      } else {
        for (const chart of EXPORT_CHARTS) {
          const node = document.querySelector(`[data-export-id="${chart.id}"]`) as HTMLElement | null;
          if (!node) continue;
          const { clientWidth, clientHeight } = node;
          if (clientWidth === 0 || clientHeight === 0) continue;

          const expandedUrl = await toPng(node, {
            cacheBust: true,
            backgroundColor: '#000000',
            width: clientWidth,
            height: clientHeight,
            pixelRatio: 2,
            style: {
              transform: 'none',
              position: 'static',
              top: '0',
              left: '0',
              right: 'auto',
              bottom: 'auto',
              margin: '0',
              width: `${clientWidth}px`,
              height: `${clientHeight}px`,
              animation: 'none',
            },
          });

          expanded[chart.id] = dataUrlToBase64(expandedUrl);
        }
      }
    }
  } finally {
    if (useFullscreenExpanded) {
      setExpandedCardForExport(null, false);
    }
  }

  return { encoding: 'base64', normal, expanded } as ExportPngs;
}
