import { Router } from 'express'
import { register, login, getMe } from './auth.controller.js'
import { validate } from '../../middleware/validate.middleware.js'
import { authenticate } from '../../middleware/auth.middleware.js'
import { registerSchema, loginSchema } from './auth.schema.js'
import rateLimit from 'express-rate-limit'

const router = Router()

// Stricter rate limit on auth endpoints — brute force protection
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 10,
  message: { error: 'Too many auth attempts. Try again in 15 minutes.' },
  standardHeaders: true,
  legacyHeaders: false,
})

router.post('/register', authLimiter, validate(registerSchema), register)
router.post('/login',    authLimiter, validate(loginSchema),    login)
router.get('/me',        authenticate,                           getMe)

export default router
