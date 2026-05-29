import { useMemo, useState } from 'react';
import type { ReactElement } from 'react';

import type { AnalyticsRow } from '../lib/types';

const ROWS: AnalyticsRow[] = [
  {
    id: 'java-entry',
    scenario: 'Java backend entry-level',
    bleu: 0.71,
    rouge: 0.69,
    latencyMs: 184,
    costPerKTokens: 0.42,
    confidence: 0.91,
    ciLow: 0.66,
    ciHigh: 0.76,
  },
  {
    id: 'frontend',
    scenario: 'Frontend engineer screening',
    bleu: 0.62,
    rouge: 0.65,
    latencyMs: 156,
    costPerKTokens: 0.37,
    confidence: 0.88,
    ciLow: 0.58,
    ciHigh: 0.68,
  },
  {
    id: 'finance',
    scenario: 'Accounts payable specialist',
    bleu: 0.82,
    rouge: 0.79,
    latencyMs: 142,
    costPerKTokens: 0.29,
    confidence: 0.95,
    ciLow: 0.77,
    ciHigh: 0.86,
  },
  {
    id: 'comparison',
    scenario: 'Assessment comparison mode',
    bleu: 0.77,
    rouge: 0.73,
    latencyMs: 218,
    costPerKTokens: 0.51,
    confidence: 0.86,
    ciLow: 0.70,
    ciHigh: 0.80,
  },
];

const FILTERS = ['all', 'fast', 'accurate', 'expensive'] as const;

type FilterKey = (typeof FILTERS)[number];

export function AnalyticsMatrix(): ReactElement {
  const [filter, setFilter] = useState<FilterKey>('all');

  const filteredRows = useMemo(() => {
    switch (filter) {
      case 'fast':
        return ROWS.filter((row) => row.latencyMs <= 160);
      case 'accurate':
        return ROWS.filter((row) => row.confidence >= 0.9);
      case 'expensive':
        return ROWS.filter((row) => row.costPerKTokens >= 0.4);
      default:
        return ROWS;
    }
  }, [filter]);

  const averages = useMemo(() => {
    const count = filteredRows.length || 1;
    return {
      bleu: filteredRows.reduce((sum, row) => sum + row.bleu, 0) / count,
      rouge: filteredRows.reduce((sum, row) => sum + row.rouge, 0) / count,
      latency: filteredRows.reduce((sum, row) => sum + row.latencyMs, 0) / count,
      cost: filteredRows.reduce((sum, row) => sum + row.costPerKTokens, 0) / count,
    };
  }, [filteredRows]);

  return (
    <section className="panel matrixShell">
      <div className="panelHeader">
        <div>
          <p className="eyebrow">Analytics</p>
          <h2>Pipeline matrix and scorecard</h2>
        </div>
        <div className="chipRow">
          {FILTERS.map((option) => (
            <button
              key={option}
              type="button"
              className={filter === option ? 'toggleButton active' : 'toggleButton'}
              onClick={() => setFilter(option)}
            >
              {option}
            </button>
          ))}
        </div>
      </div>

      <div className="summaryGrid">
        <article className="summaryCard">
          <span>Avg BLEU</span>
          <strong>{averages.bleu.toFixed(2)}</strong>
        </article>
        <article className="summaryCard">
          <span>Avg ROUGE</span>
          <strong>{averages.rouge.toFixed(2)}</strong>
        </article>
        <article className="summaryCard">
          <span>Avg latency</span>
          <strong>{averages.latency.toFixed(0)} ms</strong>
        </article>
        <article className="summaryCard">
          <span>Avg cost / 1k tokens</span>
          <strong>${averages.cost.toFixed(2)}</strong>
        </article>
      </div>

      <div className="tableWrap">
        <table className="dataTable">
          <thead>
            <tr>
              <th>Scenario</th>
              <th>BLEU</th>
              <th>ROUGE</th>
              <th>Latency</th>
              <th>Cost / 1k</th>
              <th>Confidence</th>
              <th>95% CI</th>
            </tr>
          </thead>
          <tbody>
            {filteredRows.map((row) => (
              <tr key={row.id}>
                <td>
                  <strong>{row.scenario}</strong>
                </td>
                <td>{row.bleu.toFixed(2)}</td>
                <td>{row.rouge.toFixed(2)}</td>
                <td>{row.latencyMs} ms</td>
                <td>${row.costPerKTokens.toFixed(2)}</td>
                <td>{(row.confidence * 100).toFixed(0)}%</td>
                <td>
                  {row.ciLow.toFixed(2)} - {row.ciHigh.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
