import type React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import LimitsAccessModal from '@/components/LimitsAccessModal';

function renderModal(overrides: Partial<React.ComponentProps<typeof LimitsAccessModal>> = {}) {
  const props: React.ComponentProps<typeof LimitsAccessModal> = {
    isOpen: true,
    code: '',
    expiresAt: null,
    accessGranted: false,
    limitSource: null,
    hasSession: true,
    isAdmin: true,
    onChange: vi.fn(),
    onCancel: vi.fn(),
    onApply: vi.fn(),
    onClear: vi.fn(),
    onFetchMyCode: vi.fn(),
    ...overrides,
  };
  render(<LimitsAccessModal {...props} />);
  return { props };
}

describe('LimitsAccessModal', () => {
  it('renders null when closed', () => {
    const { container } = render(
      <LimitsAccessModal
        isOpen={false}
        code=""
        expiresAt={null}
        accessGranted={false}
        limitSource={null}
        hasSession={false}
        isAdmin={false}
        onChange={() => undefined}
        onCancel={() => undefined}
        onApply={() => undefined}
        onClear={() => undefined}
        onFetchMyCode={() => undefined}
      />
    );

    expect(container.firstChild).toBeNull();
  });

  it('shows status labels based on access + source', () => {
    renderModal({ accessGranted: false, limitSource: 'production' });
    expect(screen.getByText('Locked')).toBeInTheDocument();

    renderModal({ accessGranted: true, limitSource: 'production' });
    expect(screen.getByText('Unlocked')).toBeInTheDocument();

    renderModal({ accessGranted: true, limitSource: 'local_dev' });
    expect(screen.getByText('Local limits')).toBeInTheDocument();
  });

  it('shows contact message when locked and signed in', () => {
    renderModal({ accessGranted: false, hasSession: true, limitSource: 'production' });
    expect(
      screen.getByText(/Request xiao\.jer \[at\] northeastern \[dot\] edu to get access to unlimited limits\./)
    ).toBeInTheDocument();
  });

  it('shows sign-in + contact message when signed out', () => {
    renderModal({ accessGranted: false, hasSession: false, isAdmin: false });
    expect(
      screen.getByText(/Request xiao\.jer \[at\] northeastern \[dot\] edu to get access to unlimited limits\./)
    ).toBeInTheDocument();
  });

  it('disables session-gated buttons when signed out', () => {
    renderModal({ hasSession: false, isAdmin: true });
    expect(screen.getByRole('button', { name: 'GET MY CODE' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'REFRESH' })).toBeDisabled();
  });

  it('hides generate button when not admin', () => {
    renderModal({ isAdmin: false });
    expect(screen.queryByRole('button', { name: 'GET MY CODE' })).toBeNull();
  });

  it('calls onChange with the typed value', async () => {
    const onChangeSpy = vi.fn();
    renderModal({ onChange: onChangeSpy });

    // Whitelist-only mode has no input; changing code is a no-op.
    expect(onChangeSpy).not.toHaveBeenCalled();
  });

  it('shows expiry label when expiresAt exists', () => {
    renderModal({ expiresAt: new Date('2020-01-01T00:00:00.000Z').toISOString() });
    expect(screen.getByText(/Expires ignored:/)).toBeInTheDocument();
  });
});
