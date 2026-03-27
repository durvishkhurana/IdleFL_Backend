import { Router } from 'express'
import { authenticate } from '../../middleware/auth.middleware.js'
import { downloadScript } from './agent.controller.js'

const router = Router()

router.get('/script', authenticate, downloadScript)

export default router