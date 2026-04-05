import { Router } from 'express'
import multer from 'multer'
import path from 'path'
import { authenticate } from '../../middleware/auth.middleware.js'
import { validate } from '../../middleware/validate.middleware.js'
import { createJobMultipartSchema } from './training.schema.js'
import { startTraining, abortTraining, getResults, downloadModel } from './training.controller.js'

const upload = multer({
  storage: multer.memoryStorage(),
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

router.post('/start', upload.single('dataset'), validate(createJobMultipartSchema), startTraining)
router.post('/:jobId/abort', abortTraining)
router.get('/:jobId/results', getResults)
router.get('/:jobId/model', downloadModel)

export default router
