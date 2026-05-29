export type ChatRole = 'system' | 'user' | 'assistant';

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export interface RecommendationItem {
  name: string;
  url: string;
  test_type: 'K' | 'P' | 'S' | 'C';
}

export interface ChatRequestPayload {
  messages: ChatMessage[];
}

export interface ChatResponsePayload {
  reply: string;
  recommendations: RecommendationItem[];
  end_of_conversation: boolean;
}

export interface ModelConfig {
  modelName: string;
  temperature: number;
  topP: number;
  pipeline: 'chat' | 'recommendation' | 'comparison';
  systemPrompt: string;
}

export interface AnalyticsRow {
  id: string;
  scenario: string;
  bleu: number;
  rouge: number;
  latencyMs: number;
  costPerKTokens: number;
  confidence: number;
  ciLow: number;
  ciHigh: number;
}

export interface TokenChunk {
  token: string;
  tokenId: number;
  start: number;
  end: number;
  color: string;
}
