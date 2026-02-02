'use client';

import ReactMarkdown from 'react-markdown';

interface MessageProps {
  role: 'user' | 'assistant';
  content: string;
}

export default function Message({ role, content }: MessageProps) {
  return (
    <div style={{
      padding: '24px',
      fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      <div style={{ maxWidth: '768px', margin: '0 auto', paddingLeft: '24px', paddingRight: '24px' }}>
        <div style={{ display: 'flex', gap: '16px' }}>
          {/* Avatar */}
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '50%',
            background: role === 'user' 
              ? 'linear-gradient(135deg, #6366f1, #8b5cf6)' 
              : 'linear-gradient(135deg, #10b981, #06b6d4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0
          }}>
            {role === 'user' ? (
              <span style={{ color: 'white', fontSize: '14px', fontWeight: 500 }}>Y</span>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                <path d="M2 17l10 5 10-5"/>
                <path d="M2 12l10 5 10-5"/>
              </svg>
            )}
          </div>
          
          {/* Content */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <p style={{ fontSize: '12px', fontWeight: 500, color: '#888', marginBottom: '8px' }}>
              {role === 'user' ? 'You' : 'PyReflect AI'}
            </p>
            <div className="markdown-content" style={{ color: '#e5e5e5', lineHeight: 1.7, fontSize: '15px' }}>
              <ReactMarkdown
                components={{
                  h1: ({ children }) => <h1 style={{ fontSize: '24px', fontWeight: 600, marginTop: '24px', marginBottom: '12px', color: 'white' }}>{children}</h1>,
                  h2: ({ children }) => <h2 style={{ fontSize: '20px', fontWeight: 600, marginTop: '20px', marginBottom: '10px', color: 'white' }}>{children}</h2>,
                  h3: ({ children }) => <h3 style={{ fontSize: '16px', fontWeight: 600, marginTop: '16px', marginBottom: '8px', color: 'white' }}>{children}</h3>,
                  p: ({ children }) => <p style={{ marginBottom: '12px' }}>{children}</p>,
                  ul: ({ children }) => <ul style={{ marginBottom: '12px', paddingLeft: '24px' }}>{children}</ul>,
                  ol: ({ children }) => <ol style={{ marginBottom: '12px', paddingLeft: '24px' }}>{children}</ol>,
                  li: ({ children }) => <li style={{ marginBottom: '4px' }}>{children}</li>,
                  strong: ({ children }) => <strong style={{ fontWeight: 600, color: 'white' }}>{children}</strong>,
                  em: ({ children }) => <em style={{ fontStyle: 'italic' }}>{children}</em>,
                  code: ({ children, className }) => {
                    const isBlock = className?.includes('language-');
                    if (isBlock) {
                      return (
                        <code style={{
                          display: 'block',
                          backgroundColor: '#1a1a1a',
                          padding: '16px',
                          borderRadius: '8px',
                          fontSize: '13px',
                          fontFamily: 'monospace',
                          overflowX: 'auto',
                          marginBottom: '12px'
                        }}>
                          {children}
                        </code>
                      );
                    }
                    return (
                      <code style={{
                        backgroundColor: '#333',
                        padding: '2px 6px',
                        borderRadius: '4px',
                        fontSize: '13px',
                        fontFamily: 'monospace'
                      }}>
                        {children}
                      </code>
                    );
                  },
                  pre: ({ children }) => (
                    <pre style={{
                      backgroundColor: '#1a1a1a',
                      padding: '16px',
                      borderRadius: '8px',
                      overflowX: 'auto',
                      marginBottom: '12px'
                    }}>
                      {children}
                    </pre>
                  ),
                  table: ({ children }) => (
                    <div style={{ overflowX: 'auto', marginBottom: '16px' }}>
                      <table style={{
                        width: '100%',
                        borderCollapse: 'collapse',
                        fontSize: '14px'
                      }}>
                        {children}
                      </table>
                    </div>
                  ),
                  thead: ({ children }) => <thead style={{ backgroundColor: '#2a2a2a' }}>{children}</thead>,
                  th: ({ children }) => (
                    <th style={{
                      padding: '10px 12px',
                      textAlign: 'left',
                      borderBottom: '1px solid #404040',
                      fontWeight: 600,
                      color: 'white'
                    }}>
                      {children}
                    </th>
                  ),
                  td: ({ children }) => (
                    <td style={{
                      padding: '10px 12px',
                      borderBottom: '1px solid #333'
                    }}>
                      {children}
                    </td>
                  ),
                  hr: () => <hr style={{ border: 'none', borderTop: '1px solid #404040', margin: '20px 0' }} />,
                  blockquote: ({ children }) => (
                    <blockquote style={{
                      borderLeft: '3px solid #10b981',
                      paddingLeft: '16px',
                      margin: '12px 0',
                      color: '#aaa'
                    }}>
                      {children}
                    </blockquote>
                  ),
                  a: ({ href, children }) => (
                    <a href={href} style={{ color: '#10b981', textDecoration: 'underline' }} target="_blank" rel="noopener noreferrer">
                      {children}
                    </a>
                  ),
                }}
              >
                {content}
              </ReactMarkdown>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}