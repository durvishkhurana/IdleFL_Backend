import { mkdir, writeFile } from 'fs/promises'
import path from 'path'
import { prisma } from '../../config/database.js'
import { redis, REDIS_KEYS } from '../../config/redis.js'
import { computeWeightedRoundMetrics, fedAvgAggregate } from '../../modules/training/fedavg.js'
import { getTrainingService, emitTrainingTaskAssigned } from '../../modules/training/training.service.js'
import { PARTICIPATION_THRESHOLD, ROUND_TIMEOUT_SECONDS } from '../../modules/training/training.constants.js'
import { filterEligibleDevices } from '../../utils/deviceScoring.js'
import { getShardPayload } from '../../utils/dataPartitioner.js'
import { logger } from '../../config/logger.js'

const ROUND_CACHE_TTL_SECONDS = 3600
const CHECKPOINT_TTL_SECONDS = 7200
const EVAL_WAIT_MS = 15_000
const EVAL_POLL_MS = 200
const AGG_LOCK_TTL_SECONDS = 120

const roundState = new Map()
// Key: jobId, Value: { roundNum: number, submitted: string[], totalAssigned: number, timer: any }

async function getWeightsChunkState(jobId, roundNum, deviceId) {
  const raw = await redis.get(REDIS_KEYS.jobWeightsChunk(jobId, roundNum, deviceId))
  return raw ? JSON.parse(raw) : null
}

async function saveWeightsChunkState(jobId, roundNum, deviceId, state) {
  await redis.setex(
    REDIS_KEYS.jobWeightsChunk(jobId, roundNum, deviceId),
    ROUND_CACHE_TTL_SECONDS,
    JSON.stringify(state)
  )
}

async function clearWeightsChunkState(jobId, roundNum, deviceId) {
  await redis.del(REDIS_KEYS.jobWeightsChunk(jobId, roundNum, deviceId))
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

async function getRoundContributions(jobId, roundNum) {
  const raw = await redis.get(REDIS_KEYS.jobWeights(jobId, roundNum))
  return raw ? JSON.parse(raw) : []
}

async function saveRoundContributions(jobId, roundNum, contributions) {
  await redis.setex(
    REDIS_KEYS.jobWeights(jobId, roundNum),
    ROUND_CACHE_TTL_SECONDS,
    JSON.stringify(contributions)
  )
}

async function getRoundEvalContributions(jobId, roundNum) {
  const raw = await redis.get(REDIS_KEYS.jobEval(jobId, roundNum))
  return raw ? JSON.parse(raw) : []
}

async function saveRoundEvalContributions(jobId, roundNum, contributions) {
  await redis.setex(
    REDIS_KEYS.jobEval(jobId, roundNum),
    ROUND_CACHE_TTL_SECONDS,
    JSON.stringify(contributions)
  )
}

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
 * After all assignments for the round are completed, aggregate weights, global eval, persist round, advance job.
 */
async function finalizeAggregatedRound(io, jobId, roundNum, opts = {}) {
  const lockKey = REDIS_KEYS.jobRoundAggLock(jobId, roundNum)
  const locked = await redis.set(lockKey, '1', 'EX', AGG_LOCK_TTL_SECONDS, 'NX')
  if (!locked) {
    return
  }

  const doneKey = REDIS_KEYS.jobRoundFinalized(jobId, roundNum)

  try {
    if (await redis.get(doneKey)) {
      return
    }

    if (!opts.allowEarly) {
      const inProgress = await prisma.taskAssignment.count({
        where: { jobId, roundNum, status: 'IN_PROGRESS' },
      })
      if (inProgress > 0) {
        return
      }
    } else {
      // Async aggregation: skip stragglers for this round only (do NOT mark DROPPED).
      const inProgressAssignments = await prisma.taskAssignment.findMany({
        where: { jobId, roundNum, status: 'IN_PROGRESS' },
        select: { id: true, deviceId: true, shardSize: true },
      })
      if (inProgressAssignments.length > 0) {
        logger.info(
          `Async aggregation firing for job ${jobId} round ${roundNum}: skipping ${inProgressAssignments.length} stragglers (threshold/timeout)`
        )

        const store = await getRoundContributions(jobId, roundNum)
        const seen = new Set(store.map((e) => e.deviceId))
        for (const a of inProgressAssignments) {
          if (!seen.has(a.deviceId)) {
            store.push({ deviceId: a.deviceId, skipped: true, shardSize: a.shardSize })
          }
        }
        await saveRoundContributions(jobId, roundNum, store)

        const ids = inProgressAssignments.map((a) => a.id)
        const deviceIds = inProgressAssignments.map((a) => a.deviceId)

        await prisma.taskAssignment.updateMany({
          where: { id: { in: ids } },
          data: { status: 'COMPLETED', completedAt: new Date() },
        })

        await prisma.device.updateMany({
          where: { id: { in: deviceIds } },
          data: { status: 'ACTIVE', activeTasks: 0 },
        })

        for (const deviceId of deviceIds) {
          const redisDevice = await redis.get(REDIS_KEYS.device(deviceId))
          const cachedDevice = redisDevice ? JSON.parse(redisDevice) : {}
          await redis.setex(
            REDIS_KEYS.device(deviceId),
            300,
            JSON.stringify({ ...cachedDevice, id: deviceId, status: 'ACTIVE' })
          )
          await redis.del(REDIS_KEYS.deviceTask(deviceId))
        }
      }
    }

    const job = await prisma.trainingJob.findUnique({
      where: { id: jobId },
      include: { session: true },
    })

    if (!job || job.status !== 'RUNNING') {
      return
    }

    if (job.currentRound !== roundNum) {
      return
    }

    const dedupedStore = await getRoundContributions(jobId, roundNum)
    const aggregateWeights = dedupedStore.filter(
      (c) => !c.skipped && Array.isArray(c.weights) && c.weights.length > 0
    )

    if (aggregateWeights.length === 0) {
      logger.error(`Round ${roundNum} job ${jobId}: no valid weight contributions to aggregate`)
      await prisma.trainingJob.update({ where: { id: jobId }, data: { status: 'FAILED' } })
      io.to(job.sessionId).emit('training:paused', {
        reason: 'aggregation_failed',
        message: 'No valid local models were received for this round.',
      })
      return
    }

    const { globalWeights, totalSamples } = fedAvgAggregate(aggregateWeights)
    const trainingMetrics = computeWeightedRoundMetrics(aggregateWeights)

    await redis.setex(
      REDIS_KEYS.jobGlobalWeights(jobId),
      ROUND_CACHE_TTL_SECONDS,
      JSON.stringify(globalWeights)
    )

    await redis.del(REDIS_KEYS.jobEval(jobId, roundNum))

    const contributorIds = aggregateWeights.map((e) => e.deviceId)
    const devices = await prisma.device.findMany({
      where: { id: { in: contributorIds } },
    })
    const deviceById = new Map(devices.map((d) => [d.id, d]))

    const evalPayload = {
      jobId,
      roundNum,
      globalWeights,
      modelType: job.modelType,
    }

    for (const deviceId of contributorIds) {
      const dev = deviceById.get(deviceId)
      if (dev?.socketId) {
        io.to(dev.socketId).emit('training:evaluate', evalPayload)
      }
    }

    const expectedEval = contributorIds.length
    const deadline = Date.now() + EVAL_WAIT_MS
    let evalRows = []
    while (Date.now() < deadline) {
      evalRows = await getRoundEvalContributions(jobId, roundNum)
      if (evalRows.length >= expectedEval) {
        break
      }
      await sleep(EVAL_POLL_MS)
    }

    let avgLoss
    let avgAccuracy
    if (evalRows.length > 0) {
      const m = computeWeightedRoundMetrics(evalRows)
      avgLoss = m.avgLoss
      avgAccuracy = m.avgAccuracy
    } else {
      avgLoss = trainingMetrics.avgLoss
      avgAccuracy = trainingMetrics.avgAccuracy
      logger.warn(`Round ${roundNum} job ${jobId}: no eval responses — using training-time weighted metrics`)
    }

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

    const roundDevices = await prisma.device.findMany({
      where: { id: { in: aggregateWeights.map((e) => e.deviceId) } },
      select: { id: true, deviceName: true, computeType: true, os: true },
    })
    const deviceMeta = new Map(roundDevices.map((d) => [d.id, d]))

    const assignedDevices =
      typeof opts.assignedDevices === 'number'
        ? opts.assignedDevices
        : await prisma.taskAssignment.count({ where: { jobId, roundNum } })
    const participatingDevices =
      typeof opts.participatingDevices === 'number' ? opts.participatingDevices : aggregateWeights.length

    io.to(job.sessionId).emit('training:round_complete', {
      jobId,
      round: roundNum,
      totalRounds: job.numRounds,
      loss: Number(avgLoss.toFixed(4)),
      accuracy: Number(avgAccuracy.toFixed(4)),
      participatingDevices,
      assignedDevices,
      deviceContributions: aggregateWeights.map((entry) => {
        const meta = deviceMeta.get(entry.deviceId)
        return {
          deviceId: entry.deviceId,
          name: meta?.deviceName,
          computeType: meta?.computeType,
          os: meta?.os,
          samples: entry.shardSize,
          shardSize: entry.shardSize,
          contribution: totalSamples > 0 ? entry.shardSize / totalSamples : 0,
        }
      }),
    })

    await redis.del(REDIS_KEYS.jobWeights(jobId, roundNum))
    await redis.setex(doneKey, ROUND_CACHE_TTL_SECONDS, '1')

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
  } finally {
    await redis.del(lockKey).catch(() => {})
  }
}

export async function checkRoundCompletion(io, jobId, roundNum) {
  const inProgress = await prisma.taskAssignment.count({
    where: { jobId, roundNum, status: 'IN_PROGRESS' },
  })
  if (inProgress > 0) {
    return
  }
  await finalizeAggregatedRound(io, jobId, roundNum)
}

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

  io.to(job.sessionId).emit('training:global_model', { jobId: job.id, roundNum: nextRound - 1, globalWeights })

  let mu = 0.01
  const stateRaw = await redis.get(REDIS_KEYS.jobState(job.id))
  if (stateRaw) {
    try {
      const parsed = JSON.parse(stateRaw)
      if (Number.isFinite(Number(parsed?.mu))) {
        mu = Number(parsed.mu)
      }
    } catch {
      // ignore
    }
  }

  await trainingService.dispatchRound({
    jobId: job.id,
    sessionId: job.sessionId,
    modelType: job.modelType,
    datasetPath: job.datasetPath,
    totalRows: job.totalRows,
    learningRate: job.learningRate,
    batchSize: job.batchSize,
    mu,
    roundNum: nextRound,
    globalWeights,
    devices: eligibleDevices,
  })
}

/**
 * When a device drops mid-round: reassign its IN_PROGRESS task or mark FAILED, then try to complete the round.
 */
export async function reassignDroppedDeviceTrainingTask(io, droppedDeviceId) {
  const activeTask = await prisma.taskAssignment.findFirst({
    where: { deviceId: droppedDeviceId, status: 'IN_PROGRESS' },
    include: { job: { include: { session: { include: { devices: true } } } } },
  })

  if (!activeTask || activeTask.job.status !== 'RUNNING') {
    return
  }

  const { job, roundNum, sessionId } = {
    job: activeTask.job,
    roundNum: activeTask.roundNum,
    sessionId: activeTask.job.sessionId,
  }

  if (job.currentRound !== roundNum) {
    return
  }

  const remainingDevices = activeTask.job.session.devices.filter(
    (d) => d.id !== droppedDeviceId && d.status !== 'DROPPED' && d.status !== 'DISCONNECTED'
  )

  const assignedThisRound = await prisma.taskAssignment.findMany({
    where: { jobId: activeTask.jobId, roundNum },
    select: { deviceId: true, status: true },
  })
  const busyIds = new Set(
    assignedThisRound
      .filter((a) => a.status === 'IN_PROGRESS' || a.status === 'COMPLETED')
      .map((a) => a.deviceId)
  )

  const eligible = filterEligibleDevices(remainingDevices).filter((d) => !busyIds.has(d.id))

  await prisma.taskAssignment.update({
    where: { id: activeTask.id },
    data: { status: 'REASSIGNED' },
  })

  if (eligible.length > 0) {
    const bestDevice = eligible[0]

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

    io.to(sessionId).emit('training:task_reassigned', {
      fromDeviceId: droppedDeviceId,
      toDeviceId: bestDevice.id,
      taskId: activeTask.id,
      roundNum: activeTask.roundNum,
      checkpointPath: activeTask.checkpointPath,
    })

    const redisDevice = await redis.get(REDIS_KEYS.device(bestDevice.id))
    const cachedDevice = redisDevice ? JSON.parse(redisDevice) : {}

    await prisma.device.update({
      where: { id: bestDevice.id },
      data: { status: 'TRAINING', activeTasks: 1 },
    })

    await redis.setex(
      REDIS_KEYS.device(bestDevice.id),
      300,
      JSON.stringify({
        ...cachedDevice,
        id: bestDevice.id,
        sessionId,
        socketId: bestDevice.socketId,
        status: 'TRAINING',
      })
    )

    const shard = await getShardPayload({
      datasetPath: activeTask.job.datasetPath,
      modelType: activeTask.job.modelType,
      shardStart: activeTask.shardStart,
      shardEnd: activeTask.shardEnd,
      shardSize: activeTask.shardSize,
    })

    let globalWeights = null
    const gwRaw = await redis.get(REDIS_KEYS.jobGlobalWeights(activeTask.jobId))
    if (gwRaw) {
      try {
        globalWeights = JSON.parse(gwRaw)
      } catch {
        globalWeights = null
      }
    }

    let mu = 0.01
    const stateRaw = await redis.get(REDIS_KEYS.jobState(activeTask.jobId))
    if (stateRaw) {
      try {
        const parsed = JSON.parse(stateRaw)
        if (Number.isFinite(Number(parsed?.mu))) {
          mu = Number(parsed.mu)
        }
      } catch {
        // ignore
      }
    }

    if (bestDevice.socketId) {
      emitTrainingTaskAssigned(io, {
        device: bestDevice,
        assignment: newAssignment,
        shard,
        jobId: activeTask.jobId,
        roundNum: activeTask.roundNum,
        modelType: activeTask.job.modelType,
        learningRate: activeTask.job.learningRate,
        batchSize: activeTask.job.batchSize,
        mu,
        globalWeights,
        checkpointPath: activeTask.checkpointPath,
      })
    }

    io.to(sessionId).emit('device:status_update', {
      deviceId: bestDevice.id,
      status: 'TRAINING',
    })

    logger.info(`Task ${activeTask.id} reassigned from ${droppedDeviceId} to ${bestDevice.id}`)
  } else {
    await prisma.taskAssignment.update({
      where: { id: activeTask.id },
      data: { status: 'FAILED' },
    })
    logger.warn(`No eligible device to reassign task ${activeTask.id} — marked FAILED`)
  }

  await checkRoundCompletion(io, activeTask.jobId, activeTask.roundNum)
}

export function registerTrainingHandlers(io, socket) {
  socket.on('training:weights_ready', async (payload, ack) => {
    try {
      const ackErr = (code) => {
        if (typeof ack === 'function') {
          try {
            ack({ ok: false, error: code })
          } catch {
            // ignore
          }
        }
      }

      const ackOk = (extra = {}) => {
        if (typeof ack === 'function') {
          try {
            ack({ ok: true, ...extra })
          } catch {
            // ignore
          }
        }
      }

      if (!socket.deviceId) {
        ackErr('not_registered')
        return
      }

      const {
        jobId,
        roundNum,
        weights: weightsJson,
        loss,
        accuracy,
        skipped,
        // Chunked CNN uploads (same event name, multiple emits)
        chunkIndex,
        chunkTotal,
        weightsChunk,
        // Binary one-shot uploads (preferred for CNN)
        weightsBin,
        weightsDtype,
        weightsLen,
      } = payload

      let weights = weightsJson
      const hasBinary =
        weightsBin != null &&
        (Buffer.isBuffer(weightsBin) || weightsBin instanceof ArrayBuffer || ArrayBuffer.isView(weightsBin))

      const isChunked =
        !hasBinary &&
        Number.isInteger(chunkIndex) &&
        Number.isInteger(chunkTotal) &&
        chunkTotal > 0 &&
        Array.isArray(weightsChunk)

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
        ackErr('job_not_found')
        return
      }

      if (job.status !== 'RUNNING') {
        logger.warn(`Ignoring weights for job ${jobId} in status ${job.status}`)
        ackErr('job_not_running')
        return
      }

      if (job.currentRound !== roundNum) {
        logger.warn(`Ignoring stale weights for job ${jobId}: expected round ${job.currentRound}, received ${roundNum}`)
        ackErr('stale_round')
        return
      }

      const assignment = job.taskAssignments.find((task) => task.deviceId === socket.deviceId)
      if (!assignment) {
        logger.warn(`Ignoring weights from device ${socket.deviceId}: no active assignment for round ${roundNum}`)
        ackErr('no_assignment')
        return
      }

      if (isChunked) {
        const idx = Number(chunkIndex)
        const total = Number(chunkTotal)
        if (idx < 0 || idx >= total) {
          logger.warn(`Ignoring invalid weights chunk idx=${idx}/${total} from device ${socket.deviceId}`)
          return
        }

        const existing = (await getWeightsChunkState(jobId, roundNum, socket.deviceId)) || {
          total,
          received: {},
          loss: null,
          accuracy: null,
          updatedAt: Date.now(),
        }

        if (existing.total !== total) {
          logger.warn(
            `Chunk total mismatch for job ${jobId} round ${roundNum} device ${socket.deviceId}: ` +
              `existing=${existing.total} incoming=${total} — resetting chunks`
          )
          existing.total = total
          existing.received = {}
        }

        existing.received[String(idx)] = weightsChunk
        if (loss != null) existing.loss = loss
        if (accuracy != null) existing.accuracy = accuracy
        existing.updatedAt = Date.now()

        await saveWeightsChunkState(jobId, roundNum, socket.deviceId, existing)

        const receivedCount = Object.keys(existing.received || {}).length
        logger.info(
          `Weights chunk received from device ${socket.deviceId}, job ${jobId}, round ${roundNum}: ${receivedCount}/${total}`
        )

        if (receivedCount < total) {
          ackOk({ chunked: true, receivedCount, total })
          return
        }

        const assembled = []
        for (let i = 0; i < total; i += 1) {
          const part = existing.received[String(i)]
          if (!Array.isArray(part)) {
            logger.warn(`Missing chunk ${i}/${total} for job ${jobId} round ${roundNum} device ${socket.deviceId}`)
            return
          }
          assembled.push(...part)
        }

        await clearWeightsChunkState(jobId, roundNum, socket.deviceId)

        // Continue through the normal single-payload flow with assembled weights.
        payload.weights = assembled
        payload.loss = existing.loss ?? loss
        payload.accuracy = existing.accuracy ?? accuracy
      }

      if (hasBinary) {
        const dtype = String(weightsDtype || 'float32').toLowerCase()
        if (dtype !== 'float32') {
          logger.warn(`Ignoring binary weights with unsupported dtype=${dtype} from device ${socket.deviceId}`)
          ackErr('unsupported_dtype')
          return
        }

        const buf = Buffer.isBuffer(weightsBin)
          ? weightsBin
          : Buffer.from(weightsBin.buffer || weightsBin)

        if (buf.byteLength % 4 !== 0) {
          logger.warn(
            `Ignoring binary weights with invalid byteLength=${buf.byteLength} from device ${socket.deviceId}`
          )
          ackErr('invalid_binary_length')
          return
        }

        const floatCount = buf.byteLength / 4
        if (Number.isInteger(weightsLen) && weightsLen > 0 && weightsLen !== floatCount) {
          logger.warn(
            `Binary weights length mismatch from device ${socket.deviceId}: expected=${weightsLen} got=${floatCount}`
          )
          ackErr('length_mismatch')
          return
        }

        // Avoid massive Array.from() that can block the event loop.
        // Store as base64 and decode later during aggregation.
        weights = null
        payload._weightsBinB64 = buf.toString('base64')
        payload._weightsLen = floatCount
      }

      logger.info(`Weights received from device ${socket.deviceId}, job ${jobId}, round ${roundNum}`)

      const weightStore = await getRoundContributions(jobId, roundNum)
      const dedupedStore = weightStore.filter((entry) => entry.deviceId !== socket.deviceId)

      if (skipped) {
        dedupedStore.push({
          deviceId: socket.deviceId,
          skipped: true,
          shardSize: assignment.shardSize,
        })
      } else {
        const binB64 = payload._weightsBinB64
        const binLen = payload._weightsLen
        const isBinaryStored = typeof binB64 === 'string' && binB64.length > 0 && Number.isInteger(binLen) && binLen > 0

        if (!isBinaryStored && (!Array.isArray(weights) || weights.length === 0)) {
          logger.warn(`Ignoring empty weights from device ${socket.deviceId} for job ${jobId} round ${roundNum}`)
          ackErr('empty_weights')
          return
        }

        // Ack early so agents don't time out while we write DB/Redis.
        ackOk({ received: true })

        dedupedStore.push({
          deviceId: socket.deviceId,
          ...(isBinaryStored ? { weightsBinB64: binB64, weightsLen: binLen } : { weights }),
          shardSize: assignment.shardSize,
          loss,
          accuracy,
        })
      }

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
      ackOk({
        stored: true,
        weightsLen: Array.isArray(weights) ? weights.length : payload._weightsLen || 0,
        binary: typeof payload._weightsBinB64 === 'string',
      })

      const totalAssigned = job.taskAssignments.length
      const existing = roundState.get(jobId)
      const isSameRound = existing && existing.roundNum === roundNum
      const state =
        isSameRound && existing
          ? existing
          : { roundNum, submitted: [], totalAssigned, timer: null }

      if (!isSameRound) {
        if (existing?.timer) {
          clearTimeout(existing.timer)
        }
        roundState.set(jobId, state)
      }

      if (!state.submitted.includes(socket.deviceId)) {
        state.submitted.push(socket.deviceId)
      }

      // Start timeout when first contribution arrives for the round.
      if (!state.timer) {
        state.timer = setTimeout(async () => {
          try {
            const st = roundState.get(jobId)
            if (st && st.roundNum === roundNum && st.submitted.length > 0) {
              logger.info(
                `Round ${roundNum} job ${jobId}: ROUND_TIMEOUT_SECONDS fired (${ROUND_TIMEOUT_SECONDS}s) — aggregating with ${st.submitted.length}/${st.totalAssigned}`
              )
              await finalizeAggregatedRound(io, jobId, roundNum, {
                allowEarly: true,
                participatingDevices: st.submitted.length,
                assignedDevices: st.totalAssigned,
              })
              roundState.delete(jobId)
            }
          } catch (e) {
            logger.error(`Round ${roundNum} job ${jobId}: timeout aggregation error:`, e)
          }
        }, ROUND_TIMEOUT_SECONDS * 1000)

        logger.info(
          `Round ${roundNum} job ${jobId}: started aggregation timeout timer (${ROUND_TIMEOUT_SECONDS}s); threshold=${PARTICIPATION_THRESHOLD}`
        )
      }

      const needed = Math.ceil(state.totalAssigned * PARTICIPATION_THRESHOLD)
      if (state.submitted.length >= needed) {
        logger.info(
          `Round ${roundNum} job ${jobId}: participation threshold reached (${state.submitted.length}/${state.totalAssigned} >= ${needed}) — aggregating`
        )
        if (state.timer) {
          clearTimeout(state.timer)
        }
        await finalizeAggregatedRound(io, jobId, roundNum, {
          allowEarly: true,
          participatingDevices: state.submitted.length,
          assignedDevices: state.totalAssigned,
        })
        roundState.delete(jobId)
        return
      }

      const inProgress = await prisma.taskAssignment.count({
        where: { jobId, roundNum, status: 'IN_PROGRESS' },
      })

      logger.info(`Round ${roundNum}: pending IN_PROGRESS assignments: ${inProgress}`)

      if (inProgress > 0) {
        return
      }

      await checkRoundCompletion(io, jobId, roundNum)
    } catch (error) {
      logger.error('training:weights_ready error:', error)
      if (typeof ack === 'function') {
        try {
          ack({ ok: false, error: 'server_error' })
        } catch {
          // ignore
        }
      }
    }
  })

  socket.on('training:eval_result', async (payload) => {
    try {
      if (!socket.deviceId) {
        return
      }
      const { jobId, roundNum, accuracy, loss, shardSize } = payload
      if (!jobId || roundNum === undefined) {
        return
      }

      const store = await getRoundEvalContributions(jobId, roundNum)
      const deduped = store.filter((e) => e.deviceId !== socket.deviceId)
      deduped.push({
        deviceId: socket.deviceId,
        accuracy: Number(accuracy) || 0,
        loss: Number(loss) || 0,
        shardSize: Number(shardSize) || 0,
      })
      await saveRoundEvalContributions(jobId, roundNum, deduped)
    } catch (error) {
      logger.error('training:eval_result error:', error)
    }
  })

  socket.on('training:checkpoint_fetch', async ({ checkpointPath, checkpointKey } = {}, ack) => {
    try {
      const key = checkpointKey || checkpointPath
      if (!key) {
        const payload = { weights: null, error: 'no_path' }
        if (typeof ack === 'function') ack(payload)
        socket.emit('training:checkpoint_data', payload)
        return
      }

      const raw = await redis.get(key)
      if (!raw) {
        const payload = { weights: null, error: 'not_found' }
        if (typeof ack === 'function') ack(payload)
        socket.emit('training:checkpoint_data', payload)
        logger.warn(`Checkpoint not found in Redis: ${key}`)
        return
      }

      const parsed = JSON.parse(raw)
      const weights = parsed?.weights || parsed?.checkpointData || null
      const payload = { weights }
      if (typeof ack === 'function') ack(payload)
      socket.emit('training:checkpoint_data', payload)
      logger.debug(`Checkpoint fetched for device ${socket.deviceId}: ${key}`)
    } catch (error) {
      logger.error('training:checkpoint_fetch error:', error)
      const payload = { weights: null, error: 'fetch_failed' }
      if (typeof ack === 'function') ack(payload)
      socket.emit('training:checkpoint_data', payload)
    }
  })

  socket.on('training:checkpoint', async ({ taskId, jobId, roundNum, checkpointData }) => {
    try {
      if (!socket.deviceId || !taskId || !jobId) {
        return
      }

      const checkpointKey = `checkpoint:${jobId}:${taskId}`
      await redis.setex(
        checkpointKey,
        CHECKPOINT_TTL_SECONDS,
        JSON.stringify({ deviceId: socket.deviceId, checkpointData, updatedAt: new Date().toISOString() })
      )

      await prisma.taskAssignment.update({
        where: { id: taskId },
        data: { checkpointPath: checkpointKey },
      })

      socket.emit('training:checkpoint_ack', { jobId, roundNum })
      logger.debug(`Checkpoint received: device ${socket.deviceId}, job ${jobId}, task ${taskId}, round ${roundNum}`)
    } catch (error) {
      logger.error('training:checkpoint error:', error)
    }
  })
}
