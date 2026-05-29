import { useEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent, ReactElement } from 'react';

import type { ChatMessage, ChatResponsePayload, ModelConfig, RecommendationItem } from '../lib/types';

interface ChatPlaygroundProps {
  backendStatus: string;
}

const MODEL_PRESETS: Array<{ label: string; value: string }> = [
  { label: 'Gemini 2.5 Flash', value: 'gemini-2.5-flash' },
  { label: 'Gemini 2.0 Flash', value: 'gemini-2.0-flash' },
  { label: 'Gemini Pro', value: 'gemini-pro' },
];

const PIPELINES: Array<ModelConfig['pipeline']> = ['chat', 'recommendation', 'comparison'];

const QUICK_PROMPTS = [
  'Recommend a graduate-level Java assessment.',
  'Compare OPQ32 and GSA.',
  'What do you suggest for frontend engineers?',
];

const DEFAULT_SYSTEM_PROMPT =
  'You are SHL NLP Copilot. Keep replies concise, evidence-based, and grounded in the catalog context.';

const INITIAL_ASSISTANT_MESSAGE: ChatMessage = {
  role: 'assistant',
  content:
    'Configure the model, choose a pipeline, and send a hiring requirement. The FastAPI backend will resolve the catalog response.',
};

function buildSystemMessage(config: ModelConfig): ChatMessage {
  const summary = [
    `Model: ${config.modelName}`,
    `Pipeline: ${config.pipeline}`,
    `Temperature: ${config.temperature.toFixed(2)}`,
    `Top-p: ${config.topP.toFixed(2)}`,
    `Prompt: ${config.systemPrompt.trim() || DEFAULT_SYSTEM_PROMPT}`,
  ].join('\n');

  return {
    role: 'system',
    content: summary,
  };
}

function formatRecommendation(item: RecommendationItem): string {
  return `${item.name} (${item.test_type})`;
}

function getPipelineLabel(pipeline: ModelConfig['pipeline']): string {
  switch (pipeline) {
    case 'comparison':
      return 'Comparison';
    case 'recommendation':
      return 'Recommendation';
    default:
      return 'Chat';
  }
}

export function ChatPlayground({ backendStatus }: ChatPlaygroundProps): ReactElement {
  const [messages, setMessages] = useState<ChatMessage[]>([INITIAL_ASSISTANT_MESSAGE]);
  const [draft, setDraft] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [streamingReply, setStreamingReply] = useState('');
  const [rawResponse, setRawResponse] = useState<ChatResponsePayload | null>(null);
  const [config, setConfig] = useState<ModelConfig>({
    modelName: 'gemini-2.5-flash',
    temperature: 0.2,
    topP: 0.9,
    pipeline: 'recommendation',
    systemPrompt: DEFAULT_SYSTEM_PROMPT,
  });
  const [activeFormat, setActiveFormat] = useState<'markdown' | 'json'>('markdown');
  const transcriptRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const element = transcriptRef.current;
    if (element) {
      element.scrollTop = element.scrollHeight;
    }
  }, [messages, streamingReply]);

  const systemMessage = useMemo(() => buildSystemMessage(config), [config]);

  const recommendations = rawResponse?.recommendations ?? [];

  async function animateReply(reply: string): Promise<void> {
    setStreamingReply('');

    await new Promise<void>((resolve) => {
      let index = 0;
      const step = Math.max(1, Math.ceil(reply.length / 60));
      const timer = window.setInterval(() => {
        index += step;
        if (index >= reply.length) {
          setStreamingReply(reply);
          window.clearInterval(timer);
          resolve();
          return;
        }

        setStreamingReply(reply.slice(0, index));
      }, 16);
    });
  }

  async function sendMessage(content: string): Promise<void> {
    const trimmed = content.trim();
    if (!trimmed || isSending) {
      return;
    }

    const nextMessages: ChatMessage[] = [...messages, { role: 'user', content: trimmed }];
    setMessages(nextMessages);
    setDraft('');
    setIsSending(true);
    setStreamingReply('');

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({
          messages: [systemMessage, ...nextMessages],
        }),
      });

      const payload = (await response.json()) as ChatResponsePayload | { error?: string };

      if (!response.ok) {
        const message = 'error' in payload && payload.error ? payload.error : 'The FastAPI request failed.';
        throw new Error(message);
      }

      const assistantReply = 'reply' in payload ? payload.reply : '';
      setRawResponse(payload as ChatResponsePayload);
      await animateReply(assistantReply);
      setMessages((current) => [...current, { role: 'assistant', content: assistantReply }]);
      setStreamingReply('');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unable to contact the backend.';
      setMessages((current) => [
        ...current,
        {
          role: 'assistant',
          content: `Connection error: ${message}`,
        },
      ]);
      setStreamingReply('');
    } finally {
      setIsSending(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    void sendMessage(draft);
  }

  function applyPreset(prompt: string): void {
    setDraft(prompt);
  }

  return (
    <section className="panel chatShell">
      <div className="panelHeader">
        <div>
          <p className="eyebrow">NLP Playground</p>
          <h2>FastAPI-connected conversational surface</h2>
        </div>
        <div className="statusBadge">
          <span className={`statusDot ${backendStatus === 'ok' ? 'isHealthy' : 'isDown'}`} />
          <span>{backendStatus === 'ok' ? 'FastAPI online' : 'Checking backend...'}</span>
        </div>
      </div>

      <div className="chatGrid">
        <aside className="configColumn">
          <div className="configBlock">
            <label className="fieldLabel" htmlFor="modelName">
              Model
            </label>
            <select
              id="modelName"
              className="inputControl"
              value={config.modelName}
              onChange={(event) =>
                setConfig((current) => ({ ...current, modelName: event.target.value }))
              }
            >
              {MODEL_PRESETS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div className="configBlock">
            <label className="fieldLabel" htmlFor="pipeline">
              Pipeline
            </label>
            <select
              id="pipeline"
              className="inputControl"
              value={config.pipeline}
              onChange={(event) =>
                setConfig((current) => ({
                  ...current,
                  pipeline: event.target.value as ModelConfig['pipeline'],
                }))
              }
            >
              {PIPELINES.map((pipeline) => (
                <option key={pipeline} value={pipeline}>
                  {getPipelineLabel(pipeline)}
                </option>
              ))}
            </select>
          </div>

          <div className="configBlock">
            <div className="sliderHeader">
              <label className="fieldLabel" htmlFor="temperature">
                Temperature
              </label>
              <span>{config.temperature.toFixed(2)}</span>
            </div>
            <input
              id="temperature"
              className="slider"
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={config.temperature}
              onChange={(event) =>
                setConfig((current) => ({ ...current, temperature: Number(event.target.value) }))
              }
            />
          </div>

          <div className="configBlock">
            <div className="sliderHeader">
              <label className="fieldLabel" htmlFor="topP">
                Top-p
              </label>
              <span>{config.topP.toFixed(2)}</span>
            </div>
            <input
              id="topP"
              className="slider"
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={config.topP}
              onChange={(event) =>
                setConfig((current) => ({ ...current, topP: Number(event.target.value) }))
              }
            />
          </div>

          <div className="configBlock">
            <label className="fieldLabel" htmlFor="systemPrompt">
              System prompt
            </label>
            <textarea
              id="systemPrompt"
              className="inputControl textareaControl"
              rows={8}
              value={config.systemPrompt}
              onChange={(event) =>
                setConfig((current) => ({ ...current, systemPrompt: event.target.value }))
              }
            />
          </div>

          <div className="previewCard">
            <p className="previewLabel">Current pipeline</p>
            <strong>{getPipelineLabel(config.pipeline)}</strong>
            <p className="previewSubtle">
              This summary is embedded in the system message forwarded to FastAPI.
            </p>
          </div>
        </aside>

        <div className="conversationColumn">
          <div className="conversationHeader">
            <div className="formatToggle" role="tablist" aria-label="Transcript format">
              <button
                type="button"
                className={activeFormat === 'markdown' ? 'toggleButton active' : 'toggleButton'}
                onClick={() => setActiveFormat('markdown')}
              >
                Markdown
              </button>
              <button
                type="button"
                className={activeFormat === 'json' ? 'toggleButton active' : 'toggleButton'}
                onClick={() => setActiveFormat('json')}
              >
                Raw JSON
              </button>
            </div>
            <div className="chipRow">
              <span className="chip">{messages.length} turns</span>
              <span className="chip">{recommendations.length} recommendations</span>
            </div>
          </div>

          <div className="transcript" ref={transcriptRef}>
            {messages.map((message, index) => (
              <article key={`${message.role}-${index}`} className={`messageCard ${message.role}`}>
                <div className="messageMeta">
                  <span>{message.role}</span>
                  <span>{message.role === 'user' ? 'client' : 'backend'}</span>
                </div>
                <p>{message.content}</p>
              </article>
            ))}

            {streamingReply ? (
              <article className="messageCard assistant streaming">
                <div className="messageMeta">
                  <span>assistant</span>
                  <span>streaming</span>
                </div>
                <p>{streamingReply}</p>
              </article>
            ) : null}
          </div>

          <form className="composer" onSubmit={handleSubmit}>
            <textarea
              className="inputControl composerInput"
              rows={4}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Describe the role, seniority, or comparison you want..."
            />
            <div className="composerActions">
              <div className="presetRow">
                {QUICK_PROMPTS.map((prompt) => (
                  <button key={prompt} type="button" className="presetButton" onClick={() => applyPreset(prompt)}>
                    {prompt}
                  </button>
                ))}
              </div>
              <button type="submit" className="primaryButton" disabled={isSending}>
                {isSending ? 'Sending...' : 'Send to FastAPI'}
              </button>
            </div>
          </form>
        </div>
      </div>

      <div className="insightGrid">
        <section className="insightCard">
          <h3>Assistant reply</h3>
          <div className="jsonBox">
            {activeFormat === 'json' ? (
              <pre>{JSON.stringify(rawResponse, null, 2)}</pre>
            ) : (
              <p>{rawResponse?.reply ?? 'No backend response yet.'}</p>
            )}
          </div>
        </section>

        <section className="insightCard">
          <h3>Recommendations</h3>
          {recommendations.length ? (
            <ul className="recommendationList">
              {recommendations.map((item) => (
                <li key={`${item.name}-${item.url}`} className="recommendationItem">
                  <div>
                    <strong>{formatRecommendation(item)}</strong>
                    <p>{item.url}</p>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="emptyState">Results will appear here after the FastAPI request resolves.</p>
          )}
        </section>
      </div>
    </section>
  );
}
