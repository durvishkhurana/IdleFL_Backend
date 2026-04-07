import { createReadStream } from 'fs'
import { access } from 'fs/promises'
import { constants as fsConstants } from 'fs'
import { asyncHandler } from '../../utils/asyncHandler.js'

// TrainingService is instantiated with the io instance in server.js, injected here
let trainingService

export function initTrainingController(service) {
  trainingService = service
}

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
