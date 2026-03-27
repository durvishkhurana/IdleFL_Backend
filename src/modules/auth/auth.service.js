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

  signToken(userId) {
    return jwt.sign({ userId }, process.env.JWT_SECRET, {
      expiresIn: process.env.JWT_EXPIRES_IN || '7d',
    })
  }

  async getMe(userId) {
    return prisma.user.findUnique({
      where: { id: userId },
      select: { id: true, email: true, userId: true, createdAt: true },
    })
  }
}
