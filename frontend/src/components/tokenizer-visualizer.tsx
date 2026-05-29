import { useMemo, useState } from 'react';
import type { DragEvent, ReactElement } from 'react';

import type { TokenChunk } from '../lib/types';

const SAMPLE_TEXT =
  'Java backend engineers build resilient APIs, reason about systems, and ship maintainable services.';

function hashToken(token: string): number {
  let hash = 0;
  for (let index = 0; index < token.length; index += 1) {
    hash = (hash * 31 + token.charCodeAt(index)) >>> 0;
  }
  return hash;
}

function tokenize(text: string): TokenChunk[] {
  const regex = /\S+/g;
  const chunks: TokenChunk[] = [];
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    const token = match[0];
    const tokenId = hashToken(token.toLowerCase()) % 1000;
    const hue = tokenId % 360;
    chunks.push({
      token,
      tokenId,
      start: match.index,
      end: match.index + token.length,
      color: `hsla(${hue}, 70%, 55%, 0.22)`,
    });
  }

  return chunks;
}

export function TokenizerVisualizer(): ReactElement {
  const [text, setText] = useState(SAMPLE_TEXT);
  const [dropState, setDropState] = useState<'idle' | 'active'>('idle');

  const tokens = useMemo(() => tokenize(text), [text]);

  async function handleFile(file: File | null): Promise<void> {
    if (!file) {
      return;
    }

    const content = await file.text();
    setText(content);
  }

  async function handleDrop(event: DragEvent<HTMLDivElement>): Promise<void> {
    event.preventDefault();
    setDropState('idle');
    const file = event.dataTransfer.files[0] ?? null;
    await handleFile(file);
  }

  return (
    <section className="panel tokenizerShell">
      <div className="panelHeader">
        <div>
          <p className="eyebrow">Tokenizer Visualizer</p>
          <h2>Drop text and inspect token-level boundaries</h2>
        </div>
        <div className="chipRow">
          <span className="chip">{tokens.length} tokens</span>
          <span className="chip">Highlighted by stable token ids</span>
        </div>
      </div>

      <div className="tokenizerGrid">
        <div
          className={dropState === 'active' ? 'dropzone active' : 'dropzone'}
          onDragOver={(event) => {
            event.preventDefault();
            setDropState('active');
          }}
          onDragLeave={() => setDropState('idle')}
          onDrop={handleDrop}
        >
          <p className="dropzoneTitle">Drag and drop a .txt file here</p>
          <p className="dropzoneHint">Or paste text into the editor below.</p>
          <input
            className="fileInput"
            type="file"
            accept=".txt,.md,.json,text/plain"
            onChange={(event) => void handleFile(event.target.files?.[0] ?? null)}
          />
        </div>

        <div className="tokenCanvas">
          <textarea
            className="inputControl textareaControl tokenTextarea"
            value={text}
            onChange={(event) => setText(event.target.value)}
            rows={8}
          />
          <div className="tokenStream" aria-label="Token visualization">
            {tokens.map((chunk) => (
              <span
                key={`${chunk.start}-${chunk.end}-${chunk.token}`}
                className="tokenChip"
                style={{ background: chunk.color }}
                title={`Token ID ${chunk.tokenId} • chars ${chunk.start}-${chunk.end}`}
              >
                <span className="tokenText">{chunk.token}</span>
                <span className="tokenId">#{chunk.tokenId}</span>
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
