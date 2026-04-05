
import { createReadStream } from 'fs'
import { access } from 'fs/promises'
import { constants as fsConstants } from 'fs'
import { asyncHandler } from '../../utils/asyncHandler.js'

// TrainingService is instantiated with the io instance in server.js, injected here
let trainingService

export function initTrainingController(service) {
  trainingService = service
}

function stripDatasetContent(job) {
  if (!job) return job
  const { datasetContent: _omit, ...rest } = job
  return rest
}

export const startTraining = asyncHandler(async (req, res) => {
  const { sessionId, modelType } = req.body
  const learningRate = parseFloat(req.body.learningRate)
  const numRounds = parseInt(req.body.numRounds, 10)
  const batchSize = parseInt(req.body.batchSize, 10)

  if (Number.isNaN(learningRate) || Number.isNaN(numRounds) || Number.isNaN(batchSize)) {
    return res.status(400).json({ error: 'Invalid hyperparameters' })
  }

  const file = req.file
  const demoMode = process.env.DEMO_MODE === 'true'

  let datasetContent = null
  let datasetPath = null

  if (!demoMode) {
    if (modelType === 'LINEAR_REGRESSION' || modelType === 'LOGISTIC_REGRESSION') {
      if (!file) {
        return res.status(400).json({ error: 'CSV dataset file is required for this model type' })
      }
      const ext = file.originalname.toLowerCase().endsWith('.csv')
      if (!ext) {
        return res.status(400).json({ error: 'A .csv file is required for tabular models' })
      }
      datasetContent = file.buffer.toString('utf8')
    } else if (modelType === 'CNN') {
      if (!file) {
        return res.status(400).json({ error: 'ZIP dataset file is required for CNN' })
      }
      if (!file.originalname.toLowerCase().endsWith('.zip')) {
        return res.status(400).json({ error: 'A .zip file is required for CNN' })
      }
      datasetPath = file.originalname
    }
  }

  const result = await trainingService.startTraining({
    sessionId,
    modelType,
    learningRate,
    numRounds,
    batchSize,
    datasetPath,
    datasetContent,
    userId: req.user.id,
  })

  res.status(201).json({
    job: stripDatasetContent(result.job),
    eligibleDevices: result.eligibleDevices,
  })
})

export const abortTraining = asyncHandler(async (req, res) => {
  const result = await trainingService.abortTraining(req.params.jobId, req.user.id)
  res.json(result)
})

export const getResults = asyncHandler(async (req, res) => {
  const job = await trainingService.getJobResults(req.params.jobId, req.user.id)
  res.json({ job: stripDatasetContent(job) })
})

export const downloadModel = asyncHandler(async (req, res) => {
  const job = stripDatasetContent(await trainingService.getJobResults(req.params.jobId, req.user.id))

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
  stream.pipe(res)
})
