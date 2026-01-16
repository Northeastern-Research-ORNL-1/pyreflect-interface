'use client';

type ImportNameModalProps = {
  isOpen: boolean;
  value: string;
  onChange: (value: string) => void;
  onCancel: () => void;
  onConfirm: () => void;
};

export default function ImportNameModal({
  isOpen,
  value,
  onChange,
  onCancel,
  onConfirm,
}: ImportNameModalProps) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <div className="modal-title">Name Imported Generation</div>
        <input
          type="text"
          className="control__input"
          style={{ marginTop: '8px' }}
          placeholder="e.g. Experiment 1 (max 20 chars)"
          value={value}
          maxLength={20}
          onChange={(e) => onChange(e.target.value)}
          autoFocus
          onKeyDown={(e) => e.key === 'Enter' && onConfirm()}
        />
        <div className="modal-actions">
          <button className="btn btn--outline" onClick={onCancel}>
            CANCEL
          </button>
          <button className="btn" onClick={onConfirm}>
            CONFIRM
          </button>
        </div>
      </div>
    </div>
  );
}

