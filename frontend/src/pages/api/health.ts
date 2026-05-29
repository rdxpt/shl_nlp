import type { NextApiRequest, NextApiResponse } from 'next';

import { proxyHealthCheck } from '../../lib/backend';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<{ status: string } | { error: string }>,
): Promise<void> {
  if (req.method !== 'GET') {
    res.setHeader('allow', 'GET');
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  try {
    const health = await proxyHealthCheck();
    res.status(200).json(health);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unable to reach FastAPI backend.';
    res.status(502).json({ error: message });
  }
}
