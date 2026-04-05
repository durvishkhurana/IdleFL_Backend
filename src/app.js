import express from 'express'
import helmet from 'helmet'
import cors from 'cors'
import rateLimit from 'express-rate-limit'
import { errorHandler } from './middleware/error.middleware.js'
import authRoutes from './modules/auth/auth.routes.js'
import sessionRoutes from './modules/session/session.routes.js'
import trainingRoutes from './modules/training/training.routes.js'
import agentRoutes from './modules/agent/agent.routes.js'
import { logger } from './config/logger.js'

export function createApp() {
  const app = express()
  app.set('trust proxy', 1)

  app.use(helmet())

  app.use(cors({
    origin: process.env.CLIENT_URL || 'http://localhost:5173',
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
  }))

  app.use(express.json({ limit: '10mb' }))
  app.use(express.urlencoded({ extended: true }))

  app.use(rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 500,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: 'Too many requests' },
  }))

  app.use((req, _res, next) => {
    logger.debug(`→ ${req.method} ${req.path}`)
    next()
  })

  app.use('/api/auth',     authRoutes)
  app.use('/api/sessions', sessionRoutes)
  app.use('/api/training', trainingRoutes)
  app.use('/api/agent',    agentRoutes)

  app.get('/health', (_req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString(), version: '1.0.0' })
  })

  app.use((req, res) => {
    res.status(404).json({ error: `Route ${req.method} ${req.path} not found` })
  })

  app.use(errorHandler)

  return app
}