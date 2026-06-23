import { prisma } from '../../config/database.js'
import { redis, REDIS_KEYS } from '../../config/redis.js'
import { filterEligibleDevicesAsync } from '../../utils/deviceEligibility.js'
import { partitionDataset } from '../../utils/dataPartitioner.js'
import { logger } from '../../config/logger.js'
import { config as appConfig } from '../../config/app.js'
import { epochsForModelType } from './training.constants.js'

let activeTrainingService = null

/** Avoid multi-MB globalWeights on Socket.IO (CNN round 2+ on Render). */
const CNN_SOCKET_WEIGHT_MAX = parseInt(process.env.CNN_SOCKET_WEIGHT_MAX || '5000', 10)

function packageGlobalWeightsForEmit({ modelType, jobId, globalWeights }) {
  if (
    modelType !== 'CNN' ||
    !Array.isArray(globalWeights) ||
    globalWeights.length === 0 ||
    globalWeights.length <= CNN_SOCKET_WEIGHT_MAX
  ) {
    return { globalWeights, checkpointKey: null }
  }
  return {
    globalWeights: [],
    checkpointKey: REDIS_KEYS.jobGlobalWeights(jobId),
  }
}

/**
 * @param {import('socket.io').Server} io
 * @param {Object} opts
 */
export function emitTrainingTaskAssigned(io, opts) {
  const {
    device,
    assignment,
    shard,
    jobId,
    roundNum,
    modelType,
    learningRate,
    batchSize,
    globalWeights,
    checkpointPath,
    mu,
  } = opts

  if (!device?.socketId) {
    return
  }

  const epochs = epochsForModelType(modelType)
  const shardPayload =
    modelType === 'CNN'
      ? { datasetName: shard.datasetName, indices: shard.indices, format: 'images' }
      : { shardStart: shard.shardStart, shardEnd: shard.shardEnd, shardSize: shard.shardSize, format: 'tabular' }

  const cp = checkpointPath ?? assignment?.checkpointPath ?? ''
  const packaged = packageGlobalWeightsForEmit({ modelType, jobId, globalWeights })
  const weightsForSocket = packaged.globalWeights
  const agentCheckpointKey = packaged.checkpointKey || cp || ''

  io.to(device.socketId).emit('training:task_assigned', {
    taskId: assignment.id,
    jobId,
    roundNum,
    modelType,
    checkpointKey: packaged.checkpointKey || undefined,
    config: {
      learningRate,
      batchSize,
      epochs,
      globalWeights: weightsForSocket,
      mu: mu ?? 0.01,
      proximal_mu: mu ?? 0.01,
      checkpointInterval: appConfig.checkpointInterval,
      learning_rate: learningRate,
      batch_size: batchSize,
      global_weights: weightsForSocket,
      round_num: roundNum,
      checkpointPath: agentCheckpointKey,
    },
    shard: shardPayload,
  })
}

export class TrainingService {
  constructor(io) {
    this.io = io // Socket.IO instance for emitting events
    activeTrainingService = this
  }

  async startTraining({
    sessionId,
    modelType,
    learningRate,
    numRounds,
    batchSize,
    mu,
    datasetPath,
    totalRows,
    numFeatures,
    columnNames,
    datasetHash,
    userId,
  }) {
    const parsedLearningRate = parseFloat(learningRate)
    const parsedNumRounds = parseInt(numRounds, 10)
    const parsedBatchSize = parseInt(batchSize, 10)
    const parsedMu = parseFloat(mu ?? 0.01)

    if (
      Number.isNaN(parsedLearningRate) ||
      Number.isNaN(parsedNumRounds) ||
      Number.isNaN(parsedBatchSize) ||
      Number.isNaN(parsedMu)
    ) {
      throw Object.assign(new Error('Invalid hyperparameters'), { status: 400 })
    }

    const session = await prisma.session.findUnique({
      where: { id: sessionId },
      include: { devices: true },
    })

    if (!session) throw Object.assign(new Error('Session not found'), { status: 404 })
    if (session.coordinatorId !== userId) throw Object.assign(new Error('Only coordinator can start training'), { status: 403 })

    const eligible = await filterEligibleDevicesAsync(session.devices)
    if (eligible.length === 0) {
      throw Object.assign(new Error('No eligible devices. Need at least one device with score >= 0.3'), { status: 400 })
    }

    const job = await prisma.trainingJob.create({
      data: {
        sessionId,
        userId,
        modelType,
        learningRate: parsedLearningRate,
        numRounds: parsedNumRounds,
        batchSize: parsedBatchSize,
        status: 'RUNNING',
        startedAt: new Date(),
        datasetPath,
        totalRows,
        numFeatures,
        columnNames,
        datasetHash,
        totalSamples: 1000,
      },
    })

    await redis.setex(
      REDIS_KEYS.jobState(job.id),
      3600,
      JSON.stringify({
        id: job.id,
        status: 'RUNNING',
        currentRound: 0,
        numRounds: parsedNumRounds,
        modelType,
        sessionId,
        datasetPath: job.datasetPath,
        mu: parsedMu,
      })
    )

    await prisma.session.update({ where: { id: sessionId }, data: { status: 'TRAINING' } })

    await this.dispatchRound({
      jobId: job.id,
      sessionId,
      modelType,
      datasetPath: job.datasetPath,
      totalRows: job.totalRows,
      learningRate: parsedLearningRate,
      batchSize: parsedBatchSize,
      mu: parsedMu,
      roundNum: 1,
      globalWeights: null,
      devices: eligible,
    })

    this.io.to(sessionId).emit('training:started', {
      jobId: job.id,
      modelType,
      learningRate: parsedLearningRate,
      numRounds: parsedNumRounds,
      batchSize: parsedBatchSize,
      mu: parsedMu,
    })

    logger.info(`Training started: job ${job.id}, session ${sessionId}, ${eligible.length} devices (mu=${parsedMu})`)

    return { job, eligibleDevices: eligible.length }
  }

  async completeJob(jobId, sessionId) {
    const rounds = await prisma.trainingRound.findMany({
      where: { jobId },
      orderBy: { roundNum: 'desc' },
    })

    const lastRound = rounds[0]

    await prisma.trainingJob.update({
      where: { id: jobId },
      data: {
        status: 'COMPLETED',
        completedAt: new Date(),
        finalLoss: lastRound?.loss,
        finalAccuracy: lastRound?.accuracy,
      },
    })

    await prisma.session.update({ where: { id: sessionId }, data: { status: 'ACTIVE' } })

    this.io.to(sessionId).emit('training:complete', {
      jobId,
      finalAccuracy: lastRound?.accuracy,
      finalLoss: lastRound?.loss,
      totalRounds: rounds.length,
    })

    logger.info(`Training completed: job ${jobId}`)
  }

  /**
   * @param {Object} opts
   * @param {string | null | undefined} opts.datasetPath
   * @param {number | null | undefined} opts.totalRows
   */
  async dispatchRound({
    jobId,
    sessionId,
    modelType,
    datasetPath,
    totalRows,
    learningRate,
    batchSize,
    mu,
    roundNum,
    globalWeights,
    devices,
  }) {
    const shards = await partitionDataset({ datasetPath, totalRows, devices, modelType })
    const activeDevices = devices.filter((device) => shards.some((shard) => shard.deviceId === device.id))

    await prisma.taskAssignment.deleteMany({
      where: {
        jobId,
        roundNum,
        status: { in: ['PENDING', 'IN_PROGRESS'] },
      },
    })

    const assignmentsByDeviceId = new Map()
    for (const shard of shards) {
      const assignment = await prisma.taskAssignment.create({
        data: {
          jobId,
          deviceId: shard.deviceId,
          status: 'IN_PROGRESS',
          shardStart: shard.shardStart,
          shardEnd: shard.shardEnd,
          shardSize: shard.shardSize,
          roundNum,
        },
      })
      assignmentsByDeviceId.set(shard.deviceId, assignment)
    }

    await prisma.trainingJob.update({
      where: { id: jobId },
      data: {
        currentRound: roundNum,
        totalSamples: shards.reduce((sum, shard) => sum + shard.shardSize, 0),
      },
    })

    await redis.setex(
      REDIS_KEYS.jobState(jobId),
      3600,
      JSON.stringify({
        id: jobId,
        status: 'RUNNING',
        currentRound: roundNum,
        sessionId,
        modelType,
        datasetPath,
        mu: Number.isFinite(Number(mu)) ? Number(mu) : 0.01,
      })
    )
    await redis.setex(REDIS_KEYS.jobRound(jobId), 3600, String(roundNum))

    const epochs = epochsForModelType(modelType)

    for (const device of activeDevices) {
      const shard = shards.find((item) => item.deviceId === device.id)
      const redisDevice = await redis.get(REDIS_KEYS.device(device.id))
      const cachedDevice = redisDevice ? JSON.parse(redisDevice) : {}

      await prisma.device.update({
        where: { id: device.id },
        data: { status: 'TRAINING', activeTasks: 1 },
      })

      await redis.setex(
        REDIS_KEYS.device(device.id),
        300,
        JSON.stringify({ ...cachedDevice, id: device.id, sessionId, socketId: device.socketId, status: 'TRAINING' })
      )
      await redis.setex(
        REDIS_KEYS.deviceTask(device.id),
        3600,
        JSON.stringify({ jobId, roundNum, modelType, shardStart: shard.shardStart, shardEnd: shard.shardEnd, shardSize: shard.shardSize })
      )

      this.io.to(sessionId).emit('device:status_update', { deviceId: device.id, status: 'TRAINING' })

      if (device.socketId) {
        const assignment = assignmentsByDeviceId.get(device.id)
        logger.info(
          `Round ${roundNum} dispatched — globalWeights length: ${globalWeights ? globalWeights.length : 0}, epochs: ${epochs}`
        )
        emitTrainingTaskAssigned(this.io, {
          device,
          assignment,
          shard,
          jobId,
          roundNum,
          modelType,
          learningRate,
          batchSize,
          mu,
          globalWeights,
          checkpointPath: assignment.checkpointPath,
        })
      } else {
        logger.warn(`Skipping round ${roundNum} dispatch for offline device ${device.id}`)
      }
    }

    return shards
  }

  async abortTraining(jobId, userId) {
    const job = await prisma.trainingJob.findFirst({
      where: { id: jobId, userId },
      include: { session: true },
    })

    if (!job) throw Object.assign(new Error('Job not found'), { status: 404 })

    await prisma.trainingJob.update({ where: { id: jobId }, data: { status: 'ABORTED' } })
    await prisma.session.update({ where: { id: job.sessionId }, data: { status: 'ACTIVE' } })

    this.io.to(job.sessionId).emit('training:aborted', { jobId })
    logger.info(`Training aborted: job ${jobId} by user ${userId}`)
    return { success: true }
  }

  async getJobResults(jobId, userId) {
    const job = await prisma.trainingJob.findFirst({
      where: { id: jobId, userId },
      include: {
        rounds: { orderBy: { roundNum: 'asc' } },
        taskAssignments: { include: { device: { include: { user: { select: { userId: true } } } } } },
      },
    })

    if (!job) throw Object.assign(new Error('Job not found'), { status: 404 })
    return job
  }
}

export function getTrainingService() {
  return activeTrainingService
}
