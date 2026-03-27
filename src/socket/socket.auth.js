import jwt from 'jsonwebtoken'
import { prisma } from '../config/database.js'
import { logger } from '../config/logger.js'

// Socket.IO middleware — authenticates every socket connection via JWT
export const socketAuthMiddleware = async (socket, next) => {
  try {
    const token = socket.handshake.auth?.token || socket.handshake.headers?.authorization?.slice(7)

    if (!token) {
      return next(new Error('Authentication required'))
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET)
    const user = await prisma.user.findUnique({
      where: { id: decoded.userId },
      select: { id: true, email: true, userId: true },
    })

    if (!user) return next(new Error('User not found'))

    socket.user = user
    logger.debug(`Socket authenticated: ${user.email} (${socket.id})`)
    next()
  } catch (error) {
    logger.warn(`Socket auth failed: ${error.message}`)
    next(new Error('Invalid token'))
  }
}