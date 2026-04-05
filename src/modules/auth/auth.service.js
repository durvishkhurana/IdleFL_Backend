import bcrypt from 'bcryptjs'
import jwt from 'jsonwebtoken'
import { prisma } from '../../config/database.js'
import { generateUserId } from '../../utils/sessionId.js'
import { logger } from '../../config/logger.js'

export class AuthService {
  async register({ email, password }) {
    // Check if user already exists
    const existing = await prisma.user.findUnique({ where: { email } })
    if (existing) {
      const error = new Error('Email already registered')
      error.status = 409
      throw error
    }

    const passwordHash = await bcrypt.hash(password, parseInt(process.env.BCRYPT_ROUNDS) || 10)
    const userId = generateUserId(email)

    const user = await prisma.user.create({
      data: { email, passwordHash, userId },
      select: { id: true, email: true, userId: true, createdAt: true },
    })

    const token = this.signToken(user.id)
    logger.info(`New user registered: ${email} (${userId})`)
    return { user, token }
  }

  async login({ email, password }) {
    const user = await prisma.user.findUnique({ where: { email } })
    if (!user) {
      const error = new Error('Invalid email or password')
      error.status = 401
      throw error
    }

    const valid = await bcrypt.compare(password, user.passwordHash)
    if (!valid) {
      const error = new Error('Invalid email or password')
      error.status = 401
      throw error
    }

    const { passwordHash, ...safeUser } = user
    const token = this.signToken(user.id)
    logger.info(`User logged in: ${email}`)
    return { user: safeUser, token }
  }

  async agentLogin({ agentId, sessionCode }) {
    const user = await prisma.user.findUnique({ where: { userId: agentId } })
    if (!user) {
      const error = new Error('Agent ID not found')
      error.status = 401
      throw error
    }

    const session = await prisma.session.findUnique({ where: { sessionCode } })
    if (!session) {
      const error = new Error('Session not found')
      error.status = 404
      throw error
    }

    if (session.status !== 'ACTIVE') {
      const error = new Error(`Session is not active (current status: ${session.status})`)
      error.status = 403
      throw error
    }

    if (session.expiresAt && session.expiresAt < new Date()) {
      const error = new Error('Session has expired')
      error.status = 403
      throw error
    }

    const token = this.signAgentToken(user.id, session.id)
    logger.info(`Agent login: ${agentId} joined session ${sessionCode}`)
    return { token, userId: user.userId, sessionId: session.id, role: 'agent' }
  }

  signToken(userId) {
    return jwt.sign({ userId }, process.env.JWT_SECRET, {
      expiresIn: process.env.JWT_EXPIRES_IN || '7d',
    })
  }

  signAgentToken(userId, sessionId) {
    return jwt.sign(
      { userId, sessionId, role: 'agent' },
      process.env.JWT_SECRET,
      { expiresIn: process.env.JWT_AGENT_EXPIRES_IN || '24h' }
    )
  }

  async getMe(userId) {
    return prisma.user.findUnique({
      where: { id: userId },
      select: { id: true, email: true, userId: true, createdAt: true },
    })
  }
}
