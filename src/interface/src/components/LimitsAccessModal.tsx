'use client';

type LimitsAccessModalProps = {
  isOpen: boolean;
  code: string;
  expiresAt: string | null;
  accessGranted: boolean;
  limitSource: string | null;
  hasSession: boolean;
  isAdmin: boolean;
  onChange: (value: string) => void;
  onCancel: () => void;
  onClear: () => void;
  onApply: () => void;
  onFetchMyCode: () => void;
};

export default function LimitsAccessModal({
  isOpen,
  code,
  expiresAt,
  accessGranted,
  limitSource,
  hasSession,
  isAdmin,
  onChange,
  onCancel,
  onClear,
  onApply,
  onFetchMyCode,
}: LimitsAccessModalProps) {
  if (!isOpen) return null;

  const statusLabel = accessGranted
    ? limitSource === 'local_dev'
      ? 'Local limits'
      : limitSource === 'whitelist'
        ? 'Whitelisted'
        : 'Unlocked'
    : 'Locked';

  const canInteract = hasSession;
  const showGetMyCode = isAdmin;

  return (
    <div className="modal-overlay">
      <div className="modal-content" style={{ maxWidth: '420px' }}>
        <div className="modal-title">Limits Access</div>

        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)' }}>
          Status: <span style={{ color: 'var(--text-primary)' }}>{statusLabel}</span>
          {limitSource ? ` (${limitSource})` : ''}
        </div>

        {!hasSession ? (
          <div style={{ marginTop: '12px', fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
            Request xiao.jer [at] northeastern [dot] edu to get access to unlimited limits.
          </div>
        ) : (
          <div style={{ marginTop: '14px' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '6px' }}>Access</div>

            <div style={{ fontSize: '12px', color: 'var(--text-primary)', lineHeight: 1.5 }}>
              {accessGranted
                ? 'This account is allowed to use higher limits.'
                : 'This account is not allowlisted for higher limits.'}
            </div>

            {!accessGranted && (
              <div style={{ marginTop: '10px', fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                Request xiao.jer [at] northeastern [dot] edu to get access to unlimited limits.
              </div>
            )}
          </div>
        )}

        {/* Backwards-compatible props (code/expiresAt) are kept for now, but are no-ops in whitelist-only mode. */}
        {(code || expiresAt) && (
          <div style={{ marginTop: '10px', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
            {code ? <div style={{ color: 'var(--text-muted)' }}>Code ignored (whitelist-only)</div> : null}
            {expiresAt ? <div style={{ color: 'var(--text-muted)' }}>Expires ignored: {expiresAt}</div> : null}
          </div>
        )}

        <div className="modal-actions">
          {showGetMyCode && (
            <button className="btn btn--outline" onClick={onFetchMyCode} disabled={!canInteract}>
              GET MY CODE
            </button>
          )}
          <button className="btn btn--outline" onClick={onClear} disabled={!canInteract}>
            CLEAR
          </button>
          <button className="btn btn--outline" onClick={onApply} disabled={!canInteract}>
            REFRESH
          </button>
          <button className="btn btn--outline" onClick={onCancel}>
            CLOSE
          </button>
        </div>
      </div>
    </div>
  );
}
