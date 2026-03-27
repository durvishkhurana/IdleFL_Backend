import { Server } from 'socket.io'
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
    socket.on('join:session', (sessionId) => {
      socket.join(sessionId)
      socket.sessionId = sessionId
      logger.debug(`${socket.user.email} joined room: ${sessionId}`)
    })

    socket.on('disconnect', (reason) => {
      logger.info(`Socket disconnected: ${socket.user?.email} — ${reason}`)
    })
  })

  logger.info('Socket.IO server initialized')
  return io
}