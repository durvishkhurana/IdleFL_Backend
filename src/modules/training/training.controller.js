
import { createReadStream } from 'fs'
import { access } from 'fs/promises'
import { constants as fsConstants } from 'fs'
import { asyncHandler } from '../../utils/asyncHandler.js'

// TrainingService is instantiated with the io instance in server.js, injected here
let trainingService

export function initTrainingController(service) {
  trainingService = service
}

export const uploadDataset = asyncHandler(async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' })
  }

  const { originalname, filename, size, path: filePath } = req.file
  res.json({
    message: 'Dataset uploaded successfully',
    file: { originalname, filename, size, path: filePath },
  })
})

export const startTraining = asyncHandler(async (req, res) => {
  const { sessionId, modelType, learningRate, numRounds, batchSize } = req.body
  const datasetPath = req.body.datasetPath || null

  const result = await trainingService.startTraining({
    sessionId, modelType, learningRate, numRounds, batchSize, datasetPath, userId: req.user.id,
  })

  res.status(201).json(result)
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
  stream.pipe(res)
})
