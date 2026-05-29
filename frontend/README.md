# shl_nlp Frontend

Next.js Pages Router frontend for the SHL NLP backend.

## Local Development

```bash
npm install
npm run dev
```

Set `FASTAPI_BASE_URL` in `.env.local` if you want to override the deployed backend at `https://rdxptshlnlp-dbyut.ondigitalocean.app`.

## Production Build

```bash
npm run build
npm run start
```

## Vercel Deployment

- Deploy this `frontend/` folder as the Vercel project root.
- Set `FASTAPI_BASE_URL` to the public FastAPI base URL in Vercel environment variables. The default points at `https://rdxptshlnlp-dbyut.ondigitalocean.app`.
- Keep the API routes enabled so `/api/chat` and `/api/health` proxy requests to FastAPI's `/chat` and `/health` endpoints.
