import { Router } from 'express'
import { register, login, agentLogin, getMe } from './auth.controller.js'
import { validate } from '../../middleware/validate.middleware.js'
import { authenticate } from '../../middleware/auth.middleware.js'
import { registerSchema, loginSchema, agentLoginSchema } from './auth.schema.js'
import rateLimit from 'express-rate-limit'

const router = Router()

// Stricter rate limit on human auth endpoints — brute force protection
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 10,
  message: { error: 'Too many auth attempts. Try again in 15 minutes.' },
  standardHeaders: true,
  legacyHeaders: false,
})

// Slightly looser limit for agents — they reconnect on crash/restart
const agentLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 30,
  message: { error: 'Too many agent login attempts. Try again in 15 minutes.' },
  standardHeaders: true,
  legacyHeaders: false,
})

router.post('/register',    authLimiter,  validate(registerSchema),    register)
router.post('/login',       authLimiter,  validate(loginSchema),       login)
router.post('/agent-login', agentLimiter, validate(agentLoginSchema),  agentLogin)
router.get('/me',           authenticate,                               getMe)

export default router
