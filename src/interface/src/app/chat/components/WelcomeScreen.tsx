'use client';

interface WelcomeScreenProps {
  onSuggestionClick: (suggestion: string) => void;
}

export default function WelcomeScreen({ onSuggestionClick }: WelcomeScreenProps) {
  const suggestions = [
    { title: "Analyze a sample", desc: "I have a 3-layer polymer film on silicon substrate" },
    { title: "Get SLD values", desc: "What SLD should I use for gold and titanium?" },
    { title: "Model setup", desc: "Help me set up a lipid bilayer model" },
    { title: "Learn basics", desc: "Explain how neutron reflectivity works" }
  ];

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      padding: '24px',
      fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      <h1 style={{
        fontSize: '36px',
        fontWeight: 500,
        color: 'white',
        marginBottom: '48px',
        textAlign: 'center'
      }}>
        What can I help with?
      </h1>
      
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, 1fr)',
        gap: '12px',
        maxWidth: '640px',
        width: '100%'
      }}>
        {suggestions.map((suggestion, i) => (
          <button
            key={i}
            onClick={() => onSuggestionClick(suggestion.desc)}
            style={{
              textAlign: 'left',
              padding: '16px',
              backgroundColor: '#2f2f2f',
              border: '1px solid #404040',
              borderRadius: '16px',
              cursor: 'pointer',
              transition: 'background-color 0.2s'
            }}
            onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#3a3a3a'}
            onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#2f2f2f'}
          >
            <p style={{ color: 'white', fontWeight: 500, fontSize: '14px', margin: '0 0 4px 0' }}>
              {suggestion.title}
            </p>
            <p style={{ color: '#888', fontSize: '13px', margin: 0 }}>
              {suggestion.desc}
            </p>
          </button>
        ))}
      </div>
    </div>
  );
}