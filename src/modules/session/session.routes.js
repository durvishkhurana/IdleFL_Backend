
import { Router } from 'express'
import { createSession, joinSession, getSession, getMySessions } from './session.controller.js'
import { authenticate } from '../../middleware/auth.middleware.js'
import { validate } from '../../middleware/validate.middleware.js'
import { joinSessionSchema } from './session.schema.js'

const router = Router()

router.use(authenticate) // all session routes require auth

router.post('/',     createSession)
router.post('/join', validate(joinSessionSchema), joinSession)
router.get('/',      getMySessions)
router.get('/:id',   getSession)

export default router