import Head from 'next/head';
import { useEffect, useState } from 'react';
import type { ReactElement } from 'react';

import { AnalyticsMatrix } from '../components/analytics-matrix';
import { ChatPlayground } from '../components/chat-playground';
import { TokenizerVisualizer } from '../components/tokenizer-visualizer';

const TABS = ['playground', 'analytics', 'tokenizer'] as const;

type TabKey = (typeof TABS)[number];

export default function Home(): ReactElement {
  const [activeTab, setActiveTab] = useState<TabKey>('playground');
  const [backendStatus, setBackendStatus] = useState('checking');

  useEffect(() => {
    let mounted = true;

    async function checkBackend(): Promise<void> {
      try {
        const response = await fetch('/api/health');
        if (!response.ok) {
          throw new Error('Backend not ready');
        }

        if (mounted) {
          setBackendStatus('ok');
        }
      } catch {
        if (mounted) {
          setBackendStatus('down');
        }
      }
    }

    void checkBackend();

    return () => {
      mounted = false;
    };
  }, []);

  return (
    <>
      <Head>
        <title>shl_nlp | FastAPI-connected NLP console</title>
        <meta
          name="description"
          content="A Next.js frontend connected to the SHL FastAPI backend with chat, analytics, and token visualization views."
        />
      </Head>

      <main className="appRoot">
        <div className="ambient ambientA" aria-hidden="true" />
        <div className="ambient ambientB" aria-hidden="true" />

        <section className="hero panel">
          <div>
            <p className="eyebrow">shl_nlp frontend</p>
            <h1>Technical NLP interface for FastAPI-backed assessment workflows</h1>
            <p className="heroCopy">
              This Pages Router client keeps the browser experience on Next.js while proxying all backend
              traffic through local API routes, so the FastAPI service can stay isolated.
            </p>
          </div>

          <div className="heroStats">
            <div className="heroStat">
              <span>Status</span>
              <strong>{backendStatus === 'ok' ? 'Connected' : backendStatus === 'down' ? 'Offline' : 'Checking'}</strong>
            </div>
            <div className="heroStat">
              <span>Frontend</span>
              <strong>Next.js pages router</strong>
            </div>
            <div className="heroStat">
              <span>Backend</span>
              <strong>FastAPI /chat + /health</strong>
            </div>
          </div>
        </section>

        <nav className="tabBar panel" aria-label="Primary views">
          {TABS.map((tab) => (
            <button
              key={tab}
              type="button"
              className={activeTab === tab ? 'tabButton active' : 'tabButton'}
              onClick={() => setActiveTab(tab)}
            >
              {tab}
            </button>
          ))}
        </nav>

        {activeTab === 'playground' ? <ChatPlayground backendStatus={backendStatus} /> : null}
        {activeTab === 'analytics' ? <AnalyticsMatrix /> : null}
        {activeTab === 'tokenizer' ? <TokenizerVisualizer /> : null}
      </main>
    </>
  );
}
