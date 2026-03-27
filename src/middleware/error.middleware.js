import { logger } from '../config/logger.js'

export const errorHandler = (err, req, res, next) => {
  logger.error(`${req.method} ${req.path} — ${err.message}`, {
    stack: process.env.NODE_ENV === 'development' ? err.stack : undefined,
  })

  // Prisma unique constraint violation
  if (err.code === 'P2002') {
    return res.status(409).json({ error: `${err.meta?.target} already exists` })
  }

  // Prisma record not found
  if (err.code === 'P2025') {
    return res.status(404).json({ error: 'Record not found' })
  }

  const status = err.status || err.statusCode || 500
  const message = err.message || 'Internal server error'

  res.status(status).json({
    error: message,
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
  })
}
