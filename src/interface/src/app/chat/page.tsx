'use client';

import { useState, useRef, useEffect } from 'react';
import WelcomeScreen from './components/WelcomeScreen';
import Message from './components/Message';
import ChatInput from './components/ChatInput';

interface MessageType {
  role: 'user' | 'assistant';
  content: string;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<MessageType[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async (messageText?: string) => {
    const text = messageText || input;
    if (!text.trim() || isLoading) return;

    const userMessage: MessageType = { role: 'user', content: text };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const apiKey = process.env.NEXT_PUBLIC_OPENROUTER_API_KEY;

      if (!apiKey) {
        throw new Error('API key not found');
      }

      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'openai/gpt-oss-120b',
          messages: [
            {
              role: 'system',
              content: `You are PyReflect AI, a conversational assistant that helps researchers set up neutron reflectivity experiments. You guide users step-by-step to gather sample information.

## Your Approach
- Ask ONE question at a time
- Be concise and friendly
- Wait for the user's answer before moving on
- Use their previous answers to inform follow-up questions

## Information Gathering Flow
Follow this order, asking one question per response:

1. **Substrate**: "What substrate is your sample on?" (Common: Silicon, Sapphire, Quartz, Glass)
2. **Number of layers**: "How many layers does your film have?" (excluding substrate)
3. **For each layer** (starting from substrate, going up):
   - "What material is Layer [N]?"
   - "Do you know the approximate thickness of this layer? (in Ångströms, or say 'unknown')"
4. **Environment**: "What environment will the measurement be in?" (Air, D2O, H2O, vacuum)
5. **Confirm**: Summarize all parameters and ask "Does this look correct?"

## After Gathering Info
Once confirmed, say: "Great! I'll now generate the reflectivity curves for your sample. One moment..."

Then provide the final parameters in this exact JSON format:
\`\`\`json
{
  "ready_to_generate": true,
  "substrate": "silicon",
  "layers": [
    {"name": "SiO2", "thickness": 15, "sld": 3.47, "roughness": 3},
    {"name": "Polymer A", "thickness": 100, "sld": 1.0, "roughness": 4}
  ],
  "environment": "air"
}
\`\`\`

## Common SLD Values (×10⁻⁶ Å⁻²)
- Silicon: 2.07
- SiO2: 3.47
- Air: 0
- D2O: 6.36
- H2O: -0.56
- Gold: 4.5
- Titanium: -1.95
- Polymers: 0.5-2.0 (varies)

## Rules
- NEVER dump all information at once
- NEVER show tables unless summarizing final parameters
- ALWAYS ask only ONE question per response
- Keep responses under 3 sentences until the final summary
- If user seems confused, offer examples`
            },
            ...messages.map(m => ({ role: m.role, content: m.content })),
            { role: 'user', content: text }
          ],
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error?.message || 'Request failed');
      }

      if (!data.choices?.[0]) {
        throw new Error('No response');
      }

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.choices[0].message.content,
      }]);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${errorMsg}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      backgroundColor: '#212121',
      fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      {/* Header */}
      <header style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '12px 20px',
        borderBottom: '1px solid #333'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '8px',
            background: 'linear-gradient(135deg, #10b981, #06b6d4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z"/>
              <path d="M2 17l10 5 10-5"/>
              <path d="M2 12l10 5 10-5"/>
            </svg>
          </div>
          <span style={{ color: 'white', fontWeight: 500, fontSize: '16px' }}>PyReflect AI</span>
        </div>
        <button 
          onClick={() => setMessages([])}
          style={{
            color: '#888',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          New chat
        </button>
      </header>

      {/* Main content */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        {messages.length === 0 ? (
          <WelcomeScreen onSuggestionClick={sendMessage} />
        ) : (
          <div>
            {messages.map((message, index) => (
              <Message key={index} role={message.role} content={message.content} />
            ))}
            {isLoading && (
              <div style={{ padding: '24px', maxWidth: '768px', margin: '0 auto' }}>
                <div style={{ display: 'flex', gap: '16px', paddingLeft: '24px' }}>
                  <div style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, #10b981, #06b6d4)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0
                  }}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                      <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                      <path d="M2 17l10 5 10-5"/>
                      <path d="M2 12l10 5 10-5"/>
                    </svg>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px', paddingTop: '8px' }}>
                    <div style={{ width: '8px', height: '8px', backgroundColor: '#666', borderRadius: '50%', animation: 'bounce 1s infinite' }}></div>
                    <div style={{ width: '8px', height: '8px', backgroundColor: '#666', borderRadius: '50%', animation: 'bounce 1s infinite 0.15s' }}></div>
                    <div style={{ width: '8px', height: '8px', backgroundColor: '#666', borderRadius: '50%', animation: 'bounce 1s infinite 0.3s' }}></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input at bottom */}
      <ChatInput
        onSend={sendMessage}
        disabled={isLoading}
        value={input}
        onChange={setInput}
      />
    </div>
  );
}