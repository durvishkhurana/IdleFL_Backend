/**
 * Background job: scans for devices that haven't sent a heartbeat in 90 seconds.
 * Runs every 30 seconds. Marks timed-out devices as DROPPED and emits device:disconnected to the session room.
 *
 * Fault tolerance flow:
 * 1. No heartbeat for 90s → device marked DROPPED
 * 2. Device's task retrieved from last checkpoint
 * 3. Task reassigned to highest-scoring available device
 * 4. Max work lost: 10 training iterations (checkpoint interval)
 */

import { prisma } from '../config/database.js'
import { filterEligibleDevices } from '../utils/deviceScoring.js'
import { getShardPayload } from '../utils/dataPartitioner.js'
import { logger } from '../config/logger.js'

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

    const activeTask = await prisma.taskAssignment.findFirst({
      where: { deviceId: device.id, status: 'IN_PROGRESS' },
      include: { job: { include: { session: { include: { devices: true } } } } },
    })

    if (activeTask) {
      logger.info(`Reassigning task ${activeTask.id} from dropped device ${device.id}`)

      const remainingDevices = activeTask.job.session.devices.filter(
        (d) => d.id !== device.id && d.status !== 'DROPPED' && d.status !== 'DISCONNECTED'
      )

      const eligible = filterEligibleDevices(remainingDevices)

      if (eligible.length > 0) {
        const bestDevice = eligible[0]

        await prisma.taskAssignment.update({ where: { id: activeTask.id }, data: { status: 'REASSIGNED' } })

        const newAssignment = await prisma.taskAssignment.create({
          data: {
            jobId: activeTask.jobId,
            deviceId: bestDevice.id,
            status: 'IN_PROGRESS',
            shardStart: activeTask.shardStart,
            shardEnd: activeTask.shardEnd,
            shardSize: activeTask.shardSize,
            roundNum: activeTask.roundNum,
            checkpointPath: activeTask.checkpointPath,
          },
        })

        io.to(device.sessionId).emit('training:task_reassigned', {
          fromDeviceId: device.id,
          toDeviceId: bestDevice.id,
          taskId: activeTask.id,
          roundNum: activeTask.roundNum,
          checkpointPath: activeTask.checkpointPath,
        })

        if (bestDevice.socketId) {
          await prisma.device.update({
            where: { id: bestDevice.id },
            data: { status: 'TRAINING', activeTasks: 1 },
          })

          const shard = await getShardPayload({
            datasetPath: activeTask.job.datasetPath,
            datasetContent: activeTask.job.datasetContent,
            modelType: activeTask.job.modelType,
            shardStart: activeTask.shardStart,
            shardEnd: activeTask.shardEnd,
          })

          io.to(bestDevice.socketId).emit('training:task_assigned', {
            taskId: newAssignment.id,
            jobId: activeTask.jobId,
            roundNum: activeTask.roundNum,
            modelType: activeTask.job.modelType,
            config: {
              learningRate: activeTask.job.learningRate,
              batchSize: activeTask.job.batchSize,
              epochs: 1,
              learning_rate: activeTask.job.learningRate,
              batch_size: activeTask.job.batchSize,
              checkpointPath: activeTask.checkpointPath,
            },
            shard,
          })

          io.to(device.sessionId).emit('device:status_update', {
            deviceId: bestDevice.id,
            status: 'TRAINING',
          })
        }

        logger.info(`Task reassigned to device ${bestDevice.id} (score: ${bestDevice.computeScore.toFixed(2)})`)
      } else {
        logger.warn(`No eligible devices to reassign task ${activeTask.id}`)

        await prisma.trainingJob.update({ where: { id: activeTask.jobId }, data: { status: 'PENDING' } })

        io.to(device.sessionId).emit('training:paused', {
          reason: 'no_eligible_devices',
          message: 'Training paused — no devices available for reassignment',
        })
      }
    }
  }
}
