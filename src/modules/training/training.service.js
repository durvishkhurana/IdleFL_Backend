import { prisma } from '../../config/database.js'
import { redis, REDIS_KEYS } from '../../config/redis.js'
import { filterEligibleDevices } from '../../utils/deviceScoring.js'
import { partitionDataset } from '../../utils/dataPartitioner.js'
import { fedAvgAggregate, generateMockWeights } from './fedavg.js'
import { logger } from '../../config/logger.js'
import { config } from '../../config/app.js'

let activeTrainingService = null

export class TrainingService {
  constructor(io) {
    this.io = io // Socket.IO instance for emitting events
    activeTrainingService = this
  }

  async startTraining({ sessionId, modelType, learningRate, numRounds, batchSize, datasetPath, datasetContent, userId }) {
    const parsedLearningRate = parseFloat(learningRate)
    const parsedNumRounds = parseInt(numRounds, 10)
    const parsedBatchSize = parseInt(batchSize, 10)

    if (Number.isNaN(parsedLearningRate) || Number.isNaN(parsedNumRounds) || Number.isNaN(parsedBatchSize)) {
      throw Object.assign(new Error('Invalid hyperparameters'), { status: 400 })
    }

    const session = await prisma.session.findUnique({
      where: { id: sessionId },
      include: { devices: true },
    })

    if (!session) throw Object.assign(new Error('Session not found'), { status: 404 })
    if (session.coordinatorId !== userId) throw Object.assign(new Error('Only coordinator can start training'), { status: 403 })

    const eligible = filterEligibleDevices(session.devices)
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
        datasetContent,
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
      })
    )

    await prisma.session.update({ where: { id: sessionId }, data: { status: 'TRAINING' } })

    if (process.env.DEMO_MODE !== 'true') {
      await this.dispatchRound({
        jobId: job.id,
        sessionId,
        modelType,
        datasetPath: job.datasetPath,
        datasetContent: job.datasetContent,
        learningRate: parsedLearningRate,
        batchSize: parsedBatchSize,
        roundNum: 1,
        globalWeights: null,
        devices: eligible,
      })
    }

    this.io.to(sessionId).emit('training:started', {
      jobId: job.id,
      modelType,
      learningRate: parsedLearningRate,
      numRounds: parsedNumRounds,
      batchSize: parsedBatchSize,
    })

    logger.info(`Training started: job ${job.id}, session ${sessionId}, ${eligible.length} devices`)

    if (process.env.DEMO_MODE === 'true') {
      this.runDemoTrainingLoop(job, eligible, sessionId)
    }

    return { job, eligibleDevices: eligible.length }
  }

  // Demo mode: simulates a full training run with mock weights
  async runDemoTrainingLoop(job, devices, sessionId) {
    const { numRounds, modelType } = job
    let currentRound = 0

    const tick = async () => {
      if (currentRound >= numRounds) {
        await this.completeJob(job.id, sessionId)
        return
      }

      currentRound++
      logger.info(`Demo training: job ${job.id}, round ${currentRound}/${numRounds}`)

      for (const device of devices) {
        this.io.to(sessionId).emit('device:status_update', {
          deviceId: device.id,
          status: 'TRAINING',
          round: currentRound,
        })
      }

      await new Promise((resolve) => setTimeout(resolve, 2500))

      const contributions = devices.map((d) => {
        const { weights, loss, accuracy } = generateMockWeights(modelType, currentRound)
        return { deviceId: d.id, weights, shardSize: Math.floor(1000 / devices.length), loss, accuracy }
      })

      const { globalWeights } = fedAvgAggregate(contributions)
      const avgLoss = contributions.reduce((s, c) => s + c.loss, 0) / contributions.length
      const avgAccuracy = contributions.reduce((s, c) => s + c.accuracy, 0) / contributions.length

      await prisma.trainingRound.create({
        data: { jobId: job.id, roundNum: currentRound, loss: avgLoss, accuracy: avgAccuracy, duration: 2.5 },
      })

      await prisma.trainingJob.update({ where: { id: job.id }, data: { currentRound } })

      const totalShard = contributions.reduce((s, c) => s + c.shardSize, 0)

      this.io.to(sessionId).emit('training:round_complete', {
        jobId: job.id,
        round: currentRound,
        totalRounds: numRounds,
        loss: parseFloat(avgLoss.toFixed(4)),
        accuracy: parseFloat(avgAccuracy.toFixed(4)),
        deviceContributions: devices.map((d, i) => {
          const shardSize = contributions[i]?.shardSize ?? 0
          return {
            deviceId: d.id,
            name: d.deviceName,
            computeType: d.computeType,
            os: d.os,
            samples: shardSize,
            shardSize,
            contribution: totalShard > 0 ? shardSize / totalShard : 1 / devices.length,
          }
        }),
      })

      for (const device of devices) {
        this.io.to(sessionId).emit('device:status_update', { deviceId: device.id, status: 'ACTIVE' })
      }

      setTimeout(tick, 1000)
    }

    setTimeout(tick, 1500)
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
   * Creates round assignments, caches the active task in Redis, and dispatches
   * the shard to each device socket. Rounds are synchronous in v1: every device
   * must finish before the server aggregates with FedAvg.
   *
   * @param {Object} opts
   * @param {string} opts.jobId
   * @param {string} opts.sessionId
   * @param {string} opts.modelType
   * @param {string | null} opts.datasetPath
   * @param {string | null} opts.datasetContent
   * @param {number} opts.learningRate
   * @param {number} opts.batchSize
   * @param {number} opts.roundNum
   * @param {number[] | null} opts.globalWeights
   * @param {Array} opts.devices
   * @returns {Promise<Array<Object>>}
   */
  async dispatchRound({ jobId, sessionId, modelType, datasetPath, datasetContent, learningRate, batchSize, roundNum, globalWeights, devices }) {
    const shards = await partitionDataset({ datasetPath, datasetContent, devices, modelType })
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
      })
    )
    await redis.setex(REDIS_KEYS.jobRound(jobId), 3600, String(roundNum))

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
          `Round ${roundNum} dispatched — globalWeights length: ${globalWeights ? globalWeights.length : 0}`
        )
        this.io.to(device.socketId).emit('training:task_assigned', {
          taskId: assignment.id,
          jobId,
          roundNum,
          modelType,
          config: {
            learningRate,
            batchSize,
            epochs: 1,
            globalWeights,
            checkpointInterval: config.checkpointInterval,
            learning_rate: learningRate,
            batch_size: batchSize,
            global_weights: globalWeights,
            round_num: roundNum,
          },
          shard: modelType === 'CNN'
            ? { datasetName: shard.datasetName, indices: shard.indices, format: 'images' }
            : { X: shard.X, y: shard.y, format: 'tabular' },
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
