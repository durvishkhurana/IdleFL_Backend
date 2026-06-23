# IdleFL — Backend

Coordination server for IdleFL: authentication, session & job lifecycle, device scoring, dataset sharding, **FedAvg** aggregation, and real-time orchestration over Socket.IO with heartbeat-based fault tolerance.

Part of the [IdleFL](../README.md) monorepo. For the full architecture and workflow, see **[IDLEFL.md](../IDLEFL.md)**.

---

## Tech stack

- **Node.js 20+** (ESM, `"type": "module"`)
- **Express** + Helmet + CORS + express-rate-limit
- **Socket.IO** (JWT handshake auth, tunable buffer/ping)
- **Prisma** + **PostgreSQL**
- **Redis** (ioredis) — ephemeral round state, locks, checkpoints
- **jsonwebtoken**, **bcryptjs**, **Zod**, **Winston**

---

## Project structure

```
IdleFL_Backend/
├── server.js                 # bootstrap: DB+Redis, HTTP+Socket, jobs, graceful shutdown
├── prisma/
│   ├── schema.prisma         # models + enums
│   ├── seed.js               # optional demo user
│   └── migrations/           # SQL migration history
├── scripts/
│   ├── agent_windows.py      # Windows/CUDA device agent
│   └── agent_mac.py          # macOS/MPS device agent
└── src/
    ├── app.js                # Express app (security, raw-body for CNN weights, routes)
    ├── config/               # database, redis, logger, app constants
    ├── middleware/           # JWT auth, Zod validation, error handler
    ├── modules/              # auth · session · training · agent (REST)
    ├── socket/               # Socket.IO server, handshake auth, event handlers
    ├── jobs/                 # heartbeat monitor
    └── utils/                # dataPartitioner, deviceScoring, helpers
```

---

## Setup

### Prerequisites
- Node.js 20+
- A PostgreSQL database (any provider)
- A Redis instance (any provider)

### Install & run
```bash
cp .env.example .env          # fill in the values below
npm install
npx prisma generate
npx prisma migrate dev        # apply schema
npm run dev                   # nodemon → http://localhost:4000
```

Optional demo user (`demo@idlefl.com` / `password123`):
```bash
npm run db:seed
```

### npm scripts
| Script | Action |
|---|---|
| `npm run dev` | Start with nodemon (watch + reload) |
| `npm start` | Start (production) |
| `npm run build` | `prisma generate` |
| `npm run db:migrate` | `prisma migrate dev` |
| `npm run db:studio` | Open Prisma Studio |
| `npm run db:seed` | Seed the demo user |

---

## Environment variables

Copy `.env.example` → `.env`. **Never commit a real `.env`.**

| Variable | Purpose |
|---|---|
| `PORT` | HTTP port (default 4000) |
| `NODE_ENV` | `development` / `production` |
| `CLIENT_URL` | CORS origin (frontend URL) |
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection string |
| `JWT_SECRET` | **Strong, unique secret — required in production** |
| `JWT_EXPIRES_IN`, `JWT_AGENT_EXPIRES_IN` | Token lifetimes |
| `BCRYPT_ROUNDS` | Password hashing cost |
| `UPLOAD_DIR` | Model-artifact directory (default `./uploads`) |
| `HEARTBEAT_TIMEOUT_SECONDS` | Stale-device threshold (default 90) |
| `HEARTBEAT_INTERVAL_SECONDS` | Monitor cadence (default 30) |
| `CHECKPOINT_INTERVAL` | Checkpoint cadence |
| `ROUND_TIMEOUT_SECONDS` | Aggregate partial rounds after this (default 120) |
| `DISCONNECT_GRACE_SECONDS` | Defer task reassignment after socket disconnect (default 15) |
| `BACKEND_URL` / `RENDER_EXTERNAL_URL` | Injected as agent `SERVER_URL` |
| `SOCKET_IO_MAX_HTTP_BUFFER_MB` | Large CNN payloads over Socket.IO |
| `SOCKET_IO_WEBSOCKET_ONLY` | `true` disables polling fallback |
| `SOCKET_IO_PING_TIMEOUT_MS`, `SOCKET_IO_PING_INTERVAL_MS` | Engine.IO heartbeat tuning |

---

## API summary

| Group | Endpoints |
|---|---|
| **Auth** `/api/auth` | `POST /register`, `POST /login`, `POST /agent-login`, `GET /me` |
| **Sessions** `/api/sessions` | `POST /`, `POST /join`, `GET /`, `GET /:id` |
| **Training** `/api/training` | `POST /start`, `POST /:jobId/abort`, `GET /:jobId/results`, `GET /:jobId/model`, `POST /:jobId/round/:roundNum/weights` (agent, raw binary) |
| **Agent** `/api/agent` | `GET /script?os=windows\|mac` |
| **Health** | `GET /health` |

Socket.IO events and the round lifecycle are documented in [IDLEFL.md](../IDLEFL.md#13-socketio-event-reference).

---

## Deployment notes

- Use `prisma migrate deploy` in production (`postinstall` runs `prisma generate`).
- **P3005 baselining:** for a pre-existing DB, apply the migration SQL manually, then `prisma migrate resolve --applied <migration_name>`, then use `migrate deploy`.
- Set `BACKEND_URL` / `RENDER_EXTERNAL_URL` so downloaded agents target the correct public URL.
