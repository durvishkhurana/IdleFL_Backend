import { Router } from 'express'
import { authenticate } from '../../middleware/auth.middleware.js'
import { validate } from '../../middleware/validate.middleware.js'
import { createJobSchema } from './training.schema.js'
import { startTraining, abortTraining, getResults, downloadModel, submitAgentWeights, getAgentGlobalWeights } from './training.controller.js'

const router = Router()

/** Agent CNN weights — raw binary POST (see app.js raw parser). JWT = agent token. */
router.post('/:jobId/round/:roundNum/weights', authenticate, submitAgentWeights)
router.get('/:jobId/global-weights', authenticate, getAgentGlobalWeights)

router.use(authenticate)

router.post('/start', validate(createJobSchema), startTraining)
router.post('/:jobId/abort', abortTraining)
router.get('/:jobId/results', getResults)
router.get('/:jobId/model', downloadModel)

export default router
