/**
 * Background job: scans for devices that haven't sent a heartbeat in 90 seconds.
 * Runs every 30 seconds. Marks timed-out devices as DROPPED and emits device:disconnected to the session room.
 *
 * Fault tolerance flow:
 * 1. No heartbeat for 90s → device marked DROPPED
 * 2. IN_PROGRESS training task is reassigned or marked FAILED
 * 3. checkRoundCompletion may aggregate partial rounds
 */

import { prisma } from '../config/database.js'
import { logger } from '../config/logger.js'
import { reassignDroppedDeviceTrainingTask } from '../socket/handlers/training.handler.js'
import { isDeviceInDisconnectGrace } from '../utils/deviceDisconnectGrace.js'

const HEARTBEAT_TIMEOUT_MS = (parseInt(process.env.HEARTBEAT_TIMEOUT_SECONDS) || 90) * 1000

export function startHeartbeatMonitor(io) {
  const interval = setInterval(async () => {
    try {
      await checkForTimedOutDevices(io)
    } catch (error) {
      logger.error('Heartbeat monitor error:', error)
    }
  }, 30_000)

  logger.info('✓ Heartbeat monitor started (30s interval, 90s timeout)')
  return interval
}

async function checkForTimedOutDevices(io) {
  const cutoff = new Date(Date.now() - HEARTBEAT_TIMEOUT_MS)

  const timedOut = await prisma.device.findMany({
    where: {
      lastHeartbeat: { lt: cutoff },
      status: { in: ['ACTIVE', 'TRAINING', 'IDLE'] },
    },
    include: { session: true },
  })

  if (timedOut.length === 0) return

  for (const device of timedOut) {
    if (await isDeviceInDisconnectGrace(device.id)) {
      continue
    }

    const inProgressTasks = await prisma.taskAssignment.count({
      where: { deviceId: device.id, status: 'IN_PROGRESS' },
    })
    if (inProgressTasks > 0) {
      continue
    }

    logger.warn(`Device timed out: ${device.id} (${device.deviceName}) in session ${device.session?.sessionCode}`)

    const droppedDevice = await prisma.device.update({
      where: { id: device.id },
      data: { status: 'DROPPED', socketId: null, reliabilityScore: { decrement: 0.1 } },
      select: { reliabilityScore: true },
    })
    if (droppedDevice.reliabilityScore < 0) {
      await prisma.device.update({ where: { id: device.id }, data: { reliabilityScore: 0 } })
    }

    if (device.sessionId) {
      io.to(device.sessionId).emit('device:disconnected', {
        deviceId: device.id,
        reason: 'heartbeat_timeout',
        lastSeen: device.lastHeartbeat,
      })
    }

    await reassignDroppedDeviceTrainingTask(io, device.id)
  }
}
