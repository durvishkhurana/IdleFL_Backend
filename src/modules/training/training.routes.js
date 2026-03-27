import { Router } from 'express'
import multer from 'multer'
import path from 'path'
import { authenticate } from '../../middleware/auth.middleware.js'
import { validate } from '../../middleware/validate.middleware.js'
import { createJobSchema } from './training.schema.js'
import { uploadDataset, startTraining, abortTraining, getResults, downloadModel } from './training.controller.js'

const storage = multer.diskStorage({
  destination: process.env.UPLOAD_DIR || './uploads',
  filename: (req, file, cb) => {
    const unique = `${Date.now()}-${Math.random().toString(36).slice(2)}`
    cb(null, `${unique}-${file.originalname}`)
  },
})

const upload = multer({
  storage,
  limits: { fileSize: (parseInt(process.env.MAX_FILE_SIZE_MB) || 100) * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = ['.csv', '.zip']
    const ext = path.extname(file.originalname).toLowerCase()
    if (allowed.includes(ext)) {
      cb(null, true)
    } else {
      cb(new Error('Only .csv and .zip files are allowed'))
    }
  },
})

const router = Router()

router.use(authenticate)

router.post('/upload',        upload.single('dataset'), uploadDataset)
router.post('/start',         validate(createJobSchema), startTraining)
router.post('/:jobId/abort',  abortTraining)
router.get('/:jobId/results', getResults)
router.get('/:jobId/model',   downloadModel)

export default router
