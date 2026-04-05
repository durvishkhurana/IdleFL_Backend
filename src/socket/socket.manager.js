import { Server } from 'socket.io'
import { prisma } from '../config/database.js'
import { socketAuthMiddleware } from './socket.auth.js'
import { registerDeviceHandlers } from './handlers/device.handler.js'
import { registerTrainingHandlers } from './handlers/training.handler.js'
import { registerTerminalHandlers } from './handlers/terminal.handler.js'
import { logger } from '../config/logger.js'

export function createSocketServer(httpServer) {
  const io = new Server(httpServer, {
    cors: {
      origin: process.env.CLIENT_URL || 'http://localhost:5173',
      methods: ['GET', 'POST'],
      credentials: true,
    },
    pingTimeout: 60000,
    pingInterval: 25000,
    transports: ['websocket', 'polling'],
  })

  io.use(socketAuthMiddleware)

  io.on('connection', (socket) => {
    logger.info(`Socket connected: ${socket.user.email} (${socket.id})`)

    registerDeviceHandlers(io, socket)
    registerTrainingHandlers(io, socket)
    registerTerminalHandlers(io, socket)

    // Coordinator joins their session room (from frontend)
    socket.on('join:session', async (sessionId) => {
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