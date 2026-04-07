import { Server } from 'socket.io'
import { prisma } from '../config/database.js'
import { socketAuthMiddleware } from './socket.auth.js'
import { registerDeviceHandlers } from './handlers/device.handler.js'
import { registerTrainingHandlers } from './handlers/training.handler.js'
import { registerTerminalHandlers } from './handlers/terminal.handler.js'
import { logger } from '../config/logger.js'

// CNN sends large weight tensors as JSON; default maxHttpBufferSize (~1MB) drops the socket (transport error).
const maxHttpBufferMb = parseInt(process.env.SOCKET_IO_MAX_HTTP_BUFFER_MB || '100', 10)
const maxHttpBufferSize = Math.max(1, maxHttpBufferMb) * 1024 * 1024

export function createSocketServer(httpServer) {
  const websocketOnly = String(process.env.SOCKET_IO_WEBSOCKET_ONLY || 'true').toLowerCase() === 'true'
  const io = new Server(httpServer, {
    cors: {
      origin: process.env.CLIENT_URL || 'http://localhost:5173',
      methods: ['GET', 'POST'],
      credentials: true,
    },
    // Longer pings for slow clients / heavy GPU work between heartbeats (CNN rounds).
    pingTimeout: parseInt(process.env.SOCKET_IO_PING_TIMEOUT_MS || '180000', 10),
    pingInterval: parseInt(process.env.SOCKET_IO_PING_INTERVAL_MS || '30000', 10),
    maxHttpBufferSize,
    transports: websocketOnly ? ['websocket'] : ['websocket', 'polling'],
  })

  logger.info(
    `Socket.IO: maxHttpBufferSize=${maxHttpBufferMb}MB, pingTimeout=${parseInt(process.env.SOCKET_IO_PING_TIMEOUT_MS || '180000', 10)}ms, transports=${websocketOnly ? 'websocket' : 'websocket+polling'}`
  )

  io.use(socketAuthMiddleware)

  io.on('connection', (socket) => {
    logger.info(`Socket connected: ${socket.user.email} (${socket.id})`)

    registerDeviceHandlers(io, socket)
    registerTrainingHandlers(io, socket)
    registerTerminalHandlers(io, socket)

    // Coordinator joins their session room (from frontend)
    socket.on('join:session', async (sessionId) => {
      // Idempotency: some clients emit join twice (or on reconnect).
      // Avoid duplicate logs and redundant room joins.
      if (!sessionId || typeof sessionId !== 'string') {
        return
      }
      if (socket.sessionId === sessionId) {
        return
      }
      if (socket.rooms?.has?.(sessionId)) {
        socket.sessionId = sessionId
        return
      }

      socket.join(sessionId)
      socket.sessionId = sessionId
      const session = await prisma.session.findUnique({
        where: { id: sessionId },
        select: { sessionCode: true },
      })
      const sessionCode = session?.sessionCode ?? sessionId
      logger.info(`Device joined session: ${sessionCode} — user ${socket.user.userId} socket ${socket.id}`)
    })

    socket.on('disconnect', (reason) => {
      logger.info(`Socket disconnected: ${socket.user?.email} — ${reason}`)
    })
  })

  logger.info('Socket.IO server initialized')
  return io
}