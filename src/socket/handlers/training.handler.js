import { mkdir, writeFile } from 'fs/promises'
import path from 'path'
import { prisma } from '../../config/database.js'
import { redis, REDIS_KEYS } from '../../config/redis.js'
import { fedAvgAggregate } from '../../modules/training/fedavg.js'
import { getTrainingService } from '../../modules/training/training.service.js'
import { filterEligibleDevices } from '../../utils/deviceScoring.js'
import { logger } from '../../config/logger.js'

const ROUND_CACHE_TTL_SECONDS = 3600
const CHECKPOINT_TTL_SECONDS = 7200

/**
 * Loads the current round contribution list from Redis.
 *
 * @param {string} jobId
 * @param {number} roundNum
 * @returns {Promise<Array<Object>>}
 */
async function getRoundContributions(jobId, roundNum) {
  const raw = await redis.get(REDIS_KEYS.jobWeights(jobId, roundNum))
  return raw ? JSON.parse(raw) : []
}

/**
 * Persists the current round contribution list in Redis.
 *
 * @param {string} jobId
 * @param {number} roundNum
 * @param {Array<Object>} contributions
 * @returns {Promise<void>}
 */
async function saveRoundContributions(jobId, roundNum, contributions) {
  await redis.setex(
    REDIS_KEYS.jobWeights(jobId, roundNum),
    ROUND_CACHE_TTL_SECONDS,
    JSON.stringify(contributions)
  )
}

/**
 * Saves the final global weights as a JSON artifact in the uploads directory.
 * v1 deliberately stores flat lists over JSON for transport simplicity; the
 * browser and Python agent can both consume this without tensor-specific code.
 *
 * @param {Object} params
 * @param {string} params.jobId
 * @param {string} params.modelType
 * @param {number[]} params.globalWeights
 * @param {number} params.finalAccuracy
 * @param {number} params.finalLoss
 * @param {number} params.totalRounds
 * @returns {Promise<string>}
 */
async function saveModelArtifact({ jobId, modelType, globalWeights, finalAccuracy, finalLoss, totalRounds }) {
  const uploadDir = path.resolve(process.cwd(), process.env.UPLOAD_DIR || './uploads')
  await mkdir(uploadDir, { recursive: true })

  const modelPath = path.join(uploadDir, `idlefl_model_${jobId}.json`)
  const payload = {
    globalWeights,
    modelType,
    finalAccuracy,
    finalLoss,
    totalRounds,
    completedAt: new Date().toISOString(),
  }

  await writeFile(modelPath, JSON.stringify(payload, null, 2), 'utf8')
  return modelPath
}

/**
 * Prepares the next synchronous round. FedAvg v1 intentionally waits for all
 * devices before dispatching again because the convergence story is simpler for
 * interview review and easier to reason about during live demos.
 *
 * @param {Object} params
 * @param {any} params.io
 * @param {any} params.job
 * @param {number} params.nextRound
 * @param {number[]} params.globalWeights
 * @returns {Promise<void>}
 */
async function dispatchNextRound({ io, job, nextRound, globalWeights }) {
  const trainingService = getTrainingService()
  if (!trainingService) {
    throw new Error('Training service is not initialized')
  }

  const session = await prisma.session.findUnique({
    where: { id: job.sessionId },
    include: { devices: true },
  })

  if (!session) {
    throw new Error(`Session ${job.sessionId} not found for job ${job.id}`)
  }

  const eligibleDevices = filterEligibleDevices(session.devices)
  if (eligibleDevices.length === 0) {
    await prisma.trainingJob.update({ where: { id: job.id }, data: { status: 'PENDING' } })
    io.to(job.sessionId).emit('training:paused', {
      reason: 'no_eligible_devices',
      message: 'Training paused because no eligible devices were available for the next round.',
    })
    return
  }

  await trainingService.dispatchRound({
    jobId: job.id,
    sessionId: job.sessionId,
    modelType: job.modelType,
    datasetPath: job.datasetPath,
    learningRate: job.learningRate,
    batchSize: job.batchSize,
    roundNum: nextRound,
    globalWeights,
    devices: eligibleDevices,
  })

  io.to(job.sessionId).emit('training:global_model', { jobId: job.id, roundNum: nextRound - 1, globalWeights })
}

/**
 * Registers socket handlers for device weight uploads and checkpoints.
 *
 * @param {import('socket.io').Server} io
 * @param {import('socket.io').Socket} socket
 */
export function registerTrainingHandlers(io, socket) {
  socket.on('training:weights_ready', async ({ jobId, roundNum, weights, loss, accuracy }) => {
    try {
      if (!socket.deviceId) {
        return
      }

      const job = await prisma.trainingJob.findUnique({
        where: { id: jobId },
        include: {
          session: true,
          taskAssignments: {
            where: {
              roundNum,
              status: { in: ['PENDING', 'IN_PROGRESS', 'COMPLETED'] },
            },
          },
        },
      })

      if (!job) {
        logger.warn(`Ignoring weights for missing job ${jobId}`)
        return
      }

      if (job.status !== 'RUNNING') {
        logger.warn(`Ignoring weights for job ${jobId} in status ${job.status}`)
        return
      }

      if (job.currentRound !== roundNum) {
        logger.warn(`Ignoring stale weights for job ${jobId}: expected round ${job.currentRound}, received ${roundNum}`)
        return
      }

      const assignment = job.taskAssignments.find((task) => task.deviceId === socket.deviceId)
      if (!assignment) {
        logger.warn(`Ignoring weights from device ${socket.deviceId}: no active assignment for round ${roundNum}`)
        return
      }

      logger.info(`Weights received from device ${socket.deviceId}, job ${jobId}, round ${roundNum}`)

      const weightStore = await getRoundContributions(jobId, roundNum)
      const dedupedStore = weightStore.filter((entry) => entry.deviceId !== socket.deviceId)
      dedupedStore.push({
        deviceId: socket.deviceId,
        weights,
        shardSize: assignment.shardSize,
        loss,
        accuracy,
      })
      await saveRoundContributions(jobId, roundNum, dedupedStore)

      await prisma.taskAssignment.update({
        where: { id: assignment.id },
        data: { status: 'COMPLETED', completedAt: new Date() },
      })

      await prisma.device.update({
        where: { id: socket.deviceId },
        data: { status: 'ACTIVE', activeTasks: 0 },
      })

      const redisDevice = await redis.get(REDIS_KEYS.device(socket.deviceId))
      const cachedDevice = redisDevice ? JSON.parse(redisDevice) : {}
      await redis.setex(
        REDIS_KEYS.device(socket.deviceId),
        300,
        JSON.stringify({ ...cachedDevice, id: socket.deviceId, sessionId: job.sessionId, socketId: socket.id, status: 'ACTIVE' })
      )
      await redis.del(REDIS_KEYS.deviceTask(socket.deviceId))

      io.to(job.sessionId).emit('device:status_update', { deviceId: socket.deviceId, status: 'ACTIVE' })

      const activeAssignments = await prisma.taskAssignment.findMany({
        where: {
          jobId,
          roundNum,
          status: { in: ['PENDING', 'IN_PROGRESS', 'COMPLETED'] },
        },
      })

      const expectedDeviceIds = new Set(activeAssignments.map((task) => task.deviceId))
      const receivedDeviceIds = new Set(dedupedStore.map((entry) => entry.deviceId))
      const allSubmitted = Array.from(expectedDeviceIds).every((deviceId) => receivedDeviceIds.has(deviceId))

      logger.info(`Round ${roundNum}: ${receivedDeviceIds.size}/${expectedDeviceIds.size} weights received`)

      if (!allSubmitted) {
        return
      }

      const { globalWeights, totalSamples } = fedAvgAggregate(dedupedStore)
      const avgLoss = dedupedStore.reduce((sum, entry) => sum + entry.loss, 0) / dedupedStore.length
      const avgAccuracy = dedupedStore.reduce((sum, entry) => sum + entry.accuracy, 0) / dedupedStore.length

      await prisma.trainingRound.create({
        data: {
          jobId,
          roundNum,
          loss: avgLoss,
          accuracy: avgAccuracy,
          duration: 0,
        },
      })

      await prisma.trainingJob.update({
        where: { id: jobId },
        data: {
          status: 'AGGREGATING',
          currentRound: roundNum,
          totalSamples,
        },
      })

      await redis.setex(
        REDIS_KEYS.jobState(jobId),
        ROUND_CACHE_TTL_SECONDS,
        JSON.stringify({
          id: jobId,
          status: 'AGGREGATING',
          currentRound: roundNum,
          sessionId: job.sessionId,
          modelType: job.modelType,
          datasetPath: job.datasetPath,
        })
      )

      io.to(job.sessionId).emit('training:round_complete', {
        jobId,
        round: roundNum,
        totalRounds: job.numRounds,
        loss: Number(avgLoss.toFixed(4)),
        accuracy: Number(avgAccuracy.toFixed(4)),
        deviceContributions: dedupedStore.map((entry) => ({
          deviceId: entry.deviceId,
          contribution: entry.shardSize / totalSamples,
        })),
      })

      await redis.del(REDIS_KEYS.jobWeights(jobId, roundNum))

      if (roundNum < job.numRounds) {
        await prisma.trainingJob.update({ where: { id: jobId }, data: { status: 'RUNNING' } })
        await dispatchNextRound({ io, job, nextRound: roundNum + 1, globalWeights })
        return
      }

      const modelPath = await saveModelArtifact({
        jobId,
        modelType: job.modelType,
        globalWeights,
        finalAccuracy: avgAccuracy,
        finalLoss: avgLoss,
        totalRounds: job.numRounds,
      })

      await prisma.trainingJob.update({
        where: { id: jobId },
        data: {
          modelPath,
          status: 'RUNNING',
        },
      })

      const trainingService = getTrainingService()
      if (!trainingService) {
        throw new Error('Training service is not initialized')
      }

      await trainingService.completeJob(jobId, job.sessionId)
    } catch (error) {
      logger.error('training:weights_ready error:', error)
    }
  })

  socket.on('training:checkpoint', async ({ jobId, roundNum, checkpointData }) => {
    try {
      if (!socket.deviceId) {
        return
      }

      const checkpointKey = `${REDIS_KEYS.jobWeights(jobId, roundNum)}:checkpoint`
      await redis.setex(
        checkpointKey,
        CHECKPOINT_TTL_SECONDS,
        JSON.stringify({ deviceId: socket.deviceId, checkpointData, updatedAt: new Date().toISOString() })
      )

      await prisma.taskAssignment.updateMany({
        where: {
          jobId,
          roundNum,
          deviceId: socket.deviceId,
          status: { in: ['PENDING', 'IN_PROGRESS'] },
        },
        data: { checkpointPath: checkpointKey },
      })

      socket.emit('training:checkpoint_ack', { jobId, roundNum })
      logger.debug(`Checkpoint received: device ${socket.deviceId}, job ${jobId}, round ${roundNum}`)
    } catch (error) {
      logger.error('training:checkpoint error:', error)
    }
  })
}
