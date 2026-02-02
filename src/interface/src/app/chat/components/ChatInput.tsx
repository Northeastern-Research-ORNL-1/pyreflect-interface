'use client';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled: boolean;
  value: string;
  onChange: (value: string) => void;
}

export default function ChatInput({ onSend, disabled, value, onChange }: ChatInputProps) {
  const handleSubmit = () => {
    if (value.trim() && !disabled) {
      onSend(value);
    }
  };

  return (
    <div style={{
      padding: '16px 24px 24px',
      fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      <div style={{ maxWidth: '768px', margin: '0 auto' }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          backgroundColor: '#2f2f2f',
          borderRadius: '24px',
          border: '1px solid #404040',
          padding: '4px'
        }}>
          {/* Attach button */}
          <button style={{
            padding: '12px',
            color: '#888',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 5v14M5 12h14"/>
            </svg>
          </button>
          
          {/* Input */}
          <input
            type="text"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSubmit()}
            placeholder="Ask anything"
            style={{
              flex: 1,
              backgroundColor: 'transparent',
              border: 'none',
              outline: 'none',
              padding: '12px 0',
              color: 'white',
              fontSize: '15px',
              fontFamily: 'inherit'
            }}
          />
          
          {/* Send button */}
          <button
            onClick={handleSubmit}
            disabled={disabled || !value.trim()}
            style={{
              margin: '4px',
              padding: '10px',
              borderRadius: '50%',
              border: 'none',
              cursor: value.trim() && !disabled ? 'pointer' : 'not-allowed',
              backgroundColor: value.trim() && !disabled ? 'white' : '#404040',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'background-color 0.2s'
            }}
          >
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              width="18" 
              height="18" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke={value.trim() && !disabled ? '#000' : '#666'} 
              strokeWidth="2"
            >
              <path d="M12 19V5M5 12l7-7 7 7"/>
            </svg>
          </button>
        </div>
        
        <p style={{
          fontSize: '12px',
          color: '#666',
          textAlign: 'center',
          marginTop: '12px'
        }}>
          PyReflect AI helps analyze neutron reflectivity data. Always verify parameters experimentally.
        </p>
      </div>
    </div>
  );
}