import jwt from 'jsonwebtoken'
import { prisma } from '../config/database.js'
import { logger } from '../config/logger.js'

// Attach user to req.user on protected routes
export const authenticate = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization
    if (!authHeader?.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'No token provided' })
    }

    const token = authHeader.slice(7)
    const decoded = jwt.verify(token, process.env.JWT_SECRET)

    const user = await prisma.user.findUnique({
      where: { id: decoded.userId },
      select: { id: true, email: true, userId: true },
    })

    if (!user) {
      return res.status(401).json({ error: 'User not found' })
    }

    req.user = user
    next()
  } catch (error) {
    if (error.name === 'TokenExpiredError') {
      return res.status(401).json({ error: 'Token expired' })
    }
    if (error.name === 'JsonWebTokenError') {
      return res.status(401).json({ error: 'Invalid token' })
    }
    logger.error('Auth middleware error:', error)
    next(error)
  }
}
