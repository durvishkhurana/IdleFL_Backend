import { createReadStream } from 'fs'
import { access } from 'fs/promises'
import { constants as fsConstants } from 'fs'
import { asyncHandler } from '../../utils/asyncHandler.js'
import { ingestRoundWeightsHttp } from '../../socket/handlers/training.handler.js'
import { prisma } from '../../config/database.js'
import { redis, REDIS_KEYS } from '../../config/redis.js'

// TrainingService is instantiated with the io instance in server.js, injected here
let trainingService

export function initTrainingController(service) {
  trainingService = service
}

export const submitAgentWeights = asyncHandler(async (req, res) => {
  const { jobId, roundNum } = req.params
  const round = parseInt(roundNum, 10)
  if (Number.isNaN(round)) {
    return res.status(400).json({ error: 'invalid_round' })
  }

  const loss = req.query.loss !== undefined ? parseFloat(String(req.query.loss)) : undefined
  const accuracy = req.query.accuracy !== undefined ? parseFloat(String(req.query.accuracy)) : undefined

  const bodyBuffer = Buffer.isBuffer(req.body) ? req.body : Buffer.alloc(0)

  const result = await ingestRoundWeightsHttp(trainingService.io, {
    userId: req.user.id,
    jobId,
    roundNum: round,
    bodyBuffer,
    loss: Number.isFinite(loss) ? loss : undefined,
    accuracy: Number.isFinite(accuracy) ? accuracy : undefined,
  })

  if (!result.ok) {
    return res.status(result.status).json({ error: result.error })
  }

  return res.status(200).json({ ok: true, weightsLen: result.weightsLen })
})

/** Federated global weights for round 2+ (large — use HTTP, not Socket.IO ack). */
export const getAgentGlobalWeights = asyncHandler(async (req, res) => {
  const { jobId } = req.params
  const job = await prisma.trainingJob.findUnique({
    where: { id: jobId },
    select: { id: true, sessionId: true, status: true },
  })
  if (!job) {
    return res.status(404).json({ error: 'job_not_found' })
  }

  const device = await prisma.device.findFirst({
    where: { userId: req.user.id, sessionId: job.sessionId },
    select: { id: true },
  })
  if (!device) {
    return res.status(403).json({ error: 'not_session_device' })
  }

  const raw = await redis.get(REDIS_KEYS.jobGlobalWeights(jobId))
  if (!raw) {
    return res.status(404).json({ error: 'global_weights_not_ready' })
  }

  let weights
  try {
    weights = JSON.parse(raw)
  } catch {
    return res.status(500).json({ error: 'invalid_weights_blob' })
  }
  if (!Array.isArray(weights)) {
    return res.status(500).json({ error: 'invalid_weights_format' })
  }

  return res.status(200).json({ ok: true, weights, length: weights.length })
})

export const startTraining = asyncHandler(async (req, res) => {
  const { sessionId, modelType } = req.body
  const learningRate = parseFloat(req.body.learningRate)
  const numRounds = parseInt(req.body.numRounds, 10)
  const batchSize = parseInt(req.body.batchSize, 10)
  const mu = parseFloat(req.body.mu) || 0.01

  if (Number.isNaN(learningRate) || Number.isNaN(numRounds) || Number.isNaN(batchSize)) {
    return res.status(400).json({ error: 'Invalid hyperparameters' })
  }

  // v2 privacy: CSV file is never uploaded. Frontend sends extracted metadata only.
  const datasetPath = req.body.datasetPath ?? null
  const totalRows = req.body.totalRows ?? null
  const numFeatures = req.body.numFeatures ?? null
  const columnNames = req.body.columnNames ?? null
  const datasetHash = req.body.datasetHash ?? null

  if (modelType === 'LINEAR_REGRESSION' || modelType === 'LOGISTIC_REGRESSION') {
    if (!totalRows || !numFeatures || !columnNames) {
      return res.status(400).json({ error: 'CSV metadata is required for this model type (totalRows, numFeatures, columnNames)' })
    }
  } else if (modelType === 'CNN') {
    // Frontend hardcodes MNIST metadata and sets datasetPath='mnist.zip'
    if (!datasetPath) {
      return res.status(400).json({ error: 'datasetPath is required for CNN (e.g. mnist.zip)' })
    }
  }

  const result = await trainingService.startTraining({
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
    userId: req.user.id,
  })

  res.status(201).json({
    job: result.job,
    eligibleDevices: result.eligibleDevices,
  })
})

export const abortTraining = asyncHandler(async (req, res) => {
  const result = await trainingService.abortTraining(req.params.jobId, req.user.id)
  res.json(result)
})

export const getResults = asyncHandler(async (req, res) => {
  const job = await trainingService.getJobResults(req.params.jobId, req.user.id)
  res.json({ job })
})

export const downloadModel = asyncHandler(async (req, res) => {
  const job = await trainingService.getJobResults(req.params.jobId, req.user.id)

  if (job.status !== 'COMPLETED') {
    return res.status(400).json({ error: 'Training job is not completed yet' })
  }

  if (!job.modelPath) {
    return res.status(404).json({ error: 'Model file not yet available' })
  }

  try {
    await access(job.modelPath, fsConstants.F_OK | fsConstants.R_OK)
  } catch (_error) {
    return res.status(404).json({ error: 'Model file not yet available' })
  }

  // v2: convert to real PyTorch .pt format using a Python subprocess
  res.setHeader('Content-Type', 'application/octet-stream')
  res.setHeader('Content-Disposition', `attachment; filename="idlefl_model_${job.id}.json"`)

  const stream = createReadStream(job.modelPath)
  stream.on('error', (err) => {
    if (!res.headersSent) {
      res.status(500).json({ error: 'Failed to read model file' })
    } else {
      res.destroy(err)
    }
  })
  stream.pipe(res)
})
