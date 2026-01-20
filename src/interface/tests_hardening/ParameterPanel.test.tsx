import type React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import ParameterPanel from '@/components/ParameterPanel';
import { GPU_TIERS, type GpuTier } from '@/types';

// Minimal props factory for ParameterPanel
function createDefaultProps(overrides: Partial<React.ComponentProps<typeof ParameterPanel>> = {}) {
  return {
    filmLayers: [{ name: 'Layer 1', sld: 1, isld: 0, thickness: 100, roughness: 10 }],
    generatorParams: { numCurves: 1000, numFilmLayers: 1 },
    trainingParams: {
      batchSize: 32,
      epochs: 10,
      layers: 12,
      dropout: 0,
      latentDim: 16,
      aeEpochs: 50,
      mlpEpochs: 50,
    },
    onFilmLayersChange: vi.fn(),
    onGeneratorParamsChange: vi.fn(),
    onTrainingParamsChange: vi.fn(),
    onGenerate: vi.fn(),
    onReset: vi.fn(),
    onUploadFiles: vi.fn(),
    isGenerating: false,
    isUploading: false,
    backendStatus: null,
    dataSource: 'synthetic' as const,
    workflow: 'nr_sld' as const,
    nrSldMode: 'train' as const,
    autoGenerateModelStats: true,
    onDataSourceChange: vi.fn(),
    onWorkflowChange: vi.fn(),
    onNrSldModeChange: vi.fn(),
    onAutoGenerateModelStatsChange: vi.fn(),
    ...overrides,
  };
}

describe('ParameterPanel GPU selector', () => {
  // Helper to find the GPU dropdown by looking for a select with GPU tier options
  function findGpuSelect(): HTMLSelectElement | null {
    try {
      const selects = screen.getAllByRole('combobox') as HTMLSelectElement[];
      return selects.find((s) => s.querySelector('option[value="T4"]')) || null;
    } catch {
      return null;
    }
  }

  it('renders GPU dropdown when onGpuChange is provided', () => {
    const props = createDefaultProps({
      gpu: 'T4',
      onGpuChange: vi.fn(),
    });
    render(<ParameterPanel {...props} />);

    // Should render a select dropdown with GPU tier options
    const gpuSelect = findGpuSelect();
    expect(gpuSelect).not.toBeNull();
    
    // Check all GPU tier options exist
    GPU_TIERS.forEach((tier) => {
      const option = gpuSelect?.querySelector(`option[value="${tier.value}"]`);
      expect(option).not.toBeNull();
    });
  });

  it('does not render GPU dropdown when onGpuChange is not provided', () => {
    const props = createDefaultProps({
      gpu: 'T4',
      onGpuChange: undefined,
    });
    render(<ParameterPanel {...props} />);

    // Should not render the GPU dropdown
    const gpuSelect = findGpuSelect();
    expect(gpuSelect).toBeNull();
  });

  it('shows the selected GPU value', () => {
    const props = createDefaultProps({
      gpu: 'H100',
      onGpuChange: vi.fn(),
    });
    render(<ParameterPanel {...props} />);

    const gpuSelect = findGpuSelect();
    expect(gpuSelect?.value).toBe('H100');
  });

  it('calls onGpuChange when dropdown selection changes', () => {
    const onGpuChange = vi.fn();
    const props = createDefaultProps({
      gpu: 'T4',
      onGpuChange,
    });
    render(<ParameterPanel {...props} />);

    const gpuSelect = findGpuSelect();
    expect(gpuSelect).not.toBeNull();
    fireEvent.change(gpuSelect!, { target: { value: 'A100' } });

    expect(onGpuChange).toHaveBeenCalledWith('A100');
  });

  it('displays GPU description for selected tier', () => {
    const props = createDefaultProps({
      gpu: 'B200',
      onGpuChange: vi.fn(),
    });
    render(<ParameterPanel {...props} />);

    // B200 description is "$6.25/hr, 192GB"
    expect(screen.getByText('$6.25/hr, 192GB')).toBeInTheDocument();
  });

  it('defaults gpu prop to T4 when not provided', () => {
    const onGpuChange = vi.fn();
    const props = createDefaultProps({
      onGpuChange,
    });
    render(<ParameterPanel {...props} />);

    // T4 should be selected (default)
    const gpuSelect = findGpuSelect();
    expect(gpuSelect?.value).toBe('T4');
  });
});

describe('GPU_TIERS constant', () => {
  it('has all expected GPU tiers', () => {
    const expectedValues: GpuTier[] = ['T4', 'L4', 'A10G', 'L40S', 'A100', 'A100-80GB', 'H100', 'H200', 'B200'];
    expect(GPU_TIERS.map((t) => t.value)).toEqual(expectedValues);
  });

  it('has pricing information in descriptions', () => {
    GPU_TIERS.forEach((tier) => {
      expect(tier.description).toMatch(/\$[\d.]+\/hr/);
    });
  });

  it('has VRAM information in descriptions', () => {
    GPU_TIERS.forEach((tier) => {
      expect(tier.description).toMatch(/\d+GB/);
    });
  });

  it('has speed multipliers', () => {
    GPU_TIERS.forEach((tier) => {
      expect(tier.speed).toMatch(/^\d+(\.\d+)?×$/);
    });
  });

  it('has correct pricing for each tier', () => {
    const expectedPricing: Record<GpuTier, string> = {
      T4: '$0.59/hr',
      L4: '$0.80/hr',
      A10G: '$1.10/hr',
      L40S: '$1.95/hr',
      A100: '$2.10/hr',
      'A100-80GB': '$2.50/hr',
      H100: '$3.95/hr',
      H200: '$4.54/hr',
      B200: '$6.25/hr',
    };

    GPU_TIERS.forEach((tier) => {
      expect(tier.description).toContain(expectedPricing[tier.value]);
    });
  });
});
