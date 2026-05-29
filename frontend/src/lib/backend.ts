import type { ChatRequestPayload, ChatResponsePayload } from './types';

const DEFAULT_FASTAPI_BASE_URL = 'https://rdxptshlnlp-dbyut.ondigitalocean.app';

export function getFastApiBaseUrl(): string {
  const raw = process.env.FASTAPI_BASE_URL ?? DEFAULT_FASTAPI_BASE_URL;
  return raw.replace(/\/+$/, '');
}

async function readResponseBody(response: Response): Promise<unknown> {
  const contentType = response.headers.get('content-type') ?? '';

  if (contentType.includes('application/json')) {
    return response.json();
  }

  return response.text();
}

export async function proxyChatRequest(payload: ChatRequestPayload): Promise<ChatResponsePayload> {
  const response = await fetch(`${getFastApiBaseUrl()}/chat`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  const body = await readResponseBody(response);

  if (!response.ok) {
    const message = typeof body === 'string' ? body : 'FastAPI request failed.';
    throw new Error(message);
  }

  return body as ChatResponsePayload;
}

export async function proxyHealthCheck(): Promise<{ status: string }> {
  const response = await fetch(`${getFastApiBaseUrl()}/health`, {
    method: 'GET',
    headers: {
      accept: 'application/json',
    },
  });

  const body = await readResponseBody(response);

  if (!response.ok) {
    throw new Error(typeof body === 'string' ? body : 'Health check failed.');
  }

  return body as { status: string };
}
