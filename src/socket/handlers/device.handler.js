import { prisma } from '../../config/database.js'
import { redis, REDIS_KEYS } from '../../config/redis.js'
import { scoreDevice } from '../../utils/deviceScoring.js'
import { logger } from '../../config/logger.js'
import {
  clearDeviceDisconnectGrace,
  markDeviceDisconnected,
  scheduleReassignAfterGrace,
} from '../../utils/deviceDisconnectGrace.js'
import {
  reassignDroppedDeviceTrainingTask,
  recoverStuckRoundIfNeeded,
  redeliverInProgressTaskForDevice,
} from './training.handler.js'

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

      // Pass through agent value unchanged (expected: CUDA | MPS | CPU). Default only when absent.
      const resolvedComputeType = computeType ?? 'CPU'

      // Reconnect stability: reuse the existing Device row for this user+session.
      // This avoids duplicate "live devices" entries when the socket reconnects.
      const existing = await prisma.device.findFirst({
        where: { sessionId: session.id, userId: user.id },
        select: { id: true },
      })

      const now = new Date()
      const device = existing
        ? await prisma.device.update({
            where: { id: existing.id },
            data: { socketId: socket.id, status: 'ACTIVE', os, computeType: resolvedComputeType, lastHeartbeat: now },
          })
        : await prisma.device.create({
            data: {
              sessionId: session.id,
              userId: user.id,
              deviceName: user.userId,
              os,
              computeType: resolvedComputeType,
              socketId: socket.id,
              status: 'ACTIVE',
              lastHeartbeat: now,
            },
          })

      socket.join(session.id)
      socket.deviceId = device.id
      socket.sessionId = session.id

      await clearDeviceDisconnectGrace(device.id)

      await redis.setex(
        REDIS_KEYS.device(device.id),
        300,
        JSON.stringify({ id: device.id, sessionId: session.id, userId: user.id, socketId: socket.id, computeType: device.computeType })
      )

      logger.info(`Device registered: ${user.userId} in session ${sessionCode} (${device.computeType})`)

      socket.emit('device:registered', { deviceId: device.id, sessionId: session.id })

      if (!existing) {
        io.to(session.id).emit('device:joined', {
          device: {
            id: device.id,
            deviceId: device.id,
            deviceName: device.deviceName,
            os: device.os,
            computeType: resolvedComputeType,
            status: 'ACTIVE',
            computeScore: 0.5,
          },
        })
      } else {
        io.to(session.id).emit('device:status_update', {
          deviceId: device.id,
          status: 'ACTIVE',
          os: device.os,
          computeType: resolvedComputeType,
        })

        const taskRaw = await redis.get(REDIS_KEYS.deviceTask(device.id))
        if (taskRaw) {
          try {
            const task = JSON.parse(taskRaw)
            if (task?.jobId) {
              await recoverStuckRoundIfNeeded(io, task.jobId)
            }
          } catch {
            // ignore
          }
        }

        await redeliverInProgressTaskForDevice(io, device.id)
      }
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

  // Device disconnects — defer task reassignment so agents can reconnect within the grace window.
  socket.on('disconnect', async () => {
    if (!socket.deviceId) return

    const deviceId = socket.deviceId
    const sessionId = socket.sessionId

    try {
      await markDeviceDisconnected(deviceId)

      await prisma.device.update({
        where: { id: deviceId },
        data: { status: 'DISCONNECTED', socketId: null },
      })

      if (sessionId) {
        io.to(sessionId).emit('device:disconnected', { deviceId })
      }

      logger.info(`Device disconnected: ${deviceId} (grace before reassignment)`)

      scheduleReassignAfterGrace(deviceId, async () => {
        const current = await prisma.device.findUnique({ where: { id: deviceId }, select: { socketId: true } })
        if (current?.socketId) {
          logger.info(`Device ${deviceId} reconnected before grace expired — skip reassignment`)
          return
        }
        await reassignDroppedDeviceTrainingTask(io, deviceId)
      })
    } catch (error) {
      logger.error('disconnect handler error:', error)
    }
  })
}
