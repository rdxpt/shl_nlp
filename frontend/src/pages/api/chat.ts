import type { NextApiRequest, NextApiResponse } from 'next';

import { proxyChatRequest } from '../../lib/backend';
import type { ChatResponsePayload } from '../../lib/types';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ChatResponsePayload | { error: string }>,
): Promise<void> {
  if (req.method !== 'POST') {
    res.setHeader('allow', 'POST');
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  try {
    const payload = req.body as { messages?: unknown };
    const messages = Array.isArray(payload.messages) ? payload.messages : [];
    const result = await proxyChatRequest({ messages } as never);

    res.status(200).json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unable to reach FastAPI backend.';
    res.status(502).json({ error: message });
  }
}
