import { prisma } from '../../config/database.js'
import { redis, REDIS_KEYS } from '../../config/redis.js'
import { generateSessionCode } from '../../utils/sessionId.js'
import { logger } from '../../config/logger.js'

/**
 * SessionService — creates and manages FL sessions.
 * Sessions live in Postgres; fast lookups go through Redis.
 */
export class SessionService {
  async createSession(userId) {
    // Collision-safe code generation: retry up to 10 times
    let sessionCode
    let attempts = 0
    do {
      sessionCode = generateSessionCode()
      attempts++
      if (attempts > 10) throw new Error('Could not generate unique session code')
    } while (await prisma.session.findUnique({ where: { sessionCode } }))

    const session = await prisma.session.create({
      data: {
        sessionCode,
        coordinatorId: userId,
        status: 'ACTIVE',
        expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000), // 24h TTL
      },
      include: { coordinator: { select: { email: true, userId: true } } },
    })

    // Cache in Redis so socket handlers can do instant lookups without hitting Postgres
    await redis.setex(
      REDIS_KEYS.sessionInfo(session.id),
      86400,
      JSON.stringify({ id: session.id, sessionCode, coordinatorId: userId, status: 'ACTIVE' })
    )

    logger.info(`Session created: ${sessionCode} by user ${userId}`)
    return session
  }

  async joinSession(sessionCode, userId) {
    const session = await prisma.session.findUnique({
      where: { sessionCode },
      include: {
        devices: { include: { user: { select: { email: true, userId: true } } } },
        coordinator: { select: { email: true, userId: true } },
      },
    })

    if (!session) {
      const error = new Error('Session not found')
      error.status = 404
      throw error
    }

    if (session.status === 'EXPIRED') {
      const error = new Error('Session has expired')
      error.status = 410
      throw error
    }

    return session
  }

  async getSession(sessionId, userId) {
    // Access-gated: only coordinator or enrolled devices can read session details
    const session = await prisma.session.findFirst({
      where: {
        id: sessionId,
        OR: [
          { coordinatorId: userId },
          { devices: { some: { userId } } },
        ],
      },
      include: {
        devices: { include: { user: { select: { email: true, userId: true } } } },
        coordinator: { select: { email: true, userId: true } },
        jobs: { orderBy: { createdAt: 'desc' }, take: 5 },
      },
    })

    if (!session) {
      const error = new Error('Session not found or access denied')
      error.status = 404
      throw error
    }

    return session
  }

  async getUserSessions(userId) {
    return prisma.session.findMany({
      where: {
        OR: [{ coordinatorId: userId }, { devices: { some: { userId } } }],
      },
      include: {
        _count: { select: { devices: true } },
        coordinator: { select: { email: true, userId: true } },
      },
      orderBy: { createdAt: 'desc' },
      take: 20,
    })
  }
}
