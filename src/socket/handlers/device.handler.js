import { prisma } from '../../config/database.js'
import { redis, REDIS_KEYS } from '../../config/redis.js'
import { scoreDevice } from '../../utils/deviceScoring.js'
import { logger } from '../../config/logger.js'

export function registerDeviceHandlers(io, socket) {
  const { user } = socket

  // Agent calls this when it connects and joins a session
  socket.on('device:register', async ({ sessionCode, os, computeType }) => {
    try {
      const session = await prisma.session.findUnique({ where: { sessionCode } })
      if (!session) {
        socket.emit('error', { message: 'Session not found' })
        return
      }

      const device = await prisma.device.upsert({
        where: { id: socket.handshake.auth.deviceId || 'new' },
        update: { socketId: socket.id, status: 'ACTIVE', os, computeType, lastHeartbeat: new Date() },
        create: {
          sessionId: session.id,
          userId: user.id,
          deviceName: user.userId,
          os,
          computeType,
          socketId: socket.id,
          status: 'ACTIVE',
        },
      })

      socket.join(session.id)
      socket.deviceId = device.id
      socket.sessionId = session.id

      await redis.setex(
        REDIS_KEYS.device(device.id),
        300,
        JSON.stringify({ id: device.id, sessionId: session.id, userId: user.id, socketId: socket.id, computeType })
      )

      logger.info(`Device registered: ${user.userId} in session ${sessionCode} (${computeType})`)

      socket.emit('device:registered', { deviceId: device.id, sessionId: session.id })

      io.to(session.id).emit('device:joined', {
        device: {
          id: device.id,
          deviceName: device.deviceName,
          os,
          computeType,
          status: 'ACTIVE',
          computeScore: 0.5,
        },
      })
    } catch (error) {
      logger.error('device:register error:', error)
      socket.emit('error', { message: 'Registration failed' })
    }
  })

  // Heartbeat from agent every 30 seconds
  socket.on('heartbeat', async (stats) => {
    try {
      if (!socket.deviceId) return

      const { cpuPercent, freeRamGb, totalRamGb, gpuPercent, gpuVramUsed, gpuVramTotal } = stats
      const computeScore = scoreDevice({ ...stats, computeType: stats.computeType, reliabilityScore: 1.0, activeTasks: 0 })

      await prisma.device.update({
        where: { id: socket.deviceId },
        data: {
          cpuPercent, freeRamGb, totalRamGb,
          gpuPercent: gpuPercent || 0,
          gpuVramUsed: gpuVramUsed || 0,
          gpuVramTotal: gpuVramTotal || 0,
          computeScore,
          lastHeartbeat: new Date(),
          status: 'ACTIVE',
        },
      })

      await redis.setex(REDIS_KEYS.deviceHeartbeat(socket.deviceId), 90, Date.now().toString())

      if (socket.sessionId) {
        io.to(socket.sessionId).emit('heartbeat:received', {
          deviceId: socket.deviceId,
          cpuPercent, freeRamGb, totalRamGb,
          gpuPercent: gpuPercent || 0,
          gpuVramUsed: gpuVramUsed || 0,
          gpuVramTotal: gpuVramTotal || 0,
          computeScore,
        })
      }
    } catch (error) {
      logger.error('heartbeat handler error:', error)
    }
  })

  // Device disconnects
  socket.on('disconnect', async () => {
    if (!socket.deviceId) return

    try {
      await prisma.device.update({
        where: { id: socket.deviceId },
        data: { status: 'DISCONNECTED', socketId: null },
      })

      if (socket.sessionId) {
        io.to(socket.sessionId).emit('device:disconnected', { deviceId: socket.deviceId })
      }

      logger.info(`Device disconnected: ${socket.deviceId}`)
    } catch (error) {
      logger.error('disconnect handler error:', error)
    }
  })
}