import { Router } from 'express'
import { authenticate } from '../../middleware/auth.middleware.js'
import { validate } from '../../middleware/validate.middleware.js'
import { createJobSchema } from './training.schema.js'
import { startTraining, abortTraining, getResults, downloadModel } from './training.controller.js'

const router = Router()

router.use(authenticate)

router.post('/start', validate(createJobSchema), startTraining)
router.post('/:jobId/abort', abortTraining)
router.get('/:jobId/results', getResults)
router.get('/:jobId/model', downloadModel)

export default router
