import { PrismaClient } from '@prisma/client'
import { logger } from './logger.js'

const globalForPrisma = global

export const prisma = globalForPrisma.prisma ?? new PrismaClient({
  log: [
    { emit: 'event', level: 'query' },
    { emit: 'event', level: 'error' },
    { emit: 'event', level: 'warn' },
  ],
})

if (process.env.NODE_ENV === 'development') {
  prisma.$on('query', (e) => {
    logger.debug(`Query: ${e.query} — ${e.duration}ms`)
  })
}

prisma.$on('error', (e) => {
  logger.error('Prisma error:', e)
})

if (process.env.NODE_ENV !== 'production') {
  globalForPrisma.prisma = prisma
}

export async function connectDatabase() {
  try {
    await prisma.$connect()
    logger.info('✓ PostgreSQL connected via Prisma')
  } catch (error) {
    logger.error('✗ PostgreSQL connection failed:', error)
    process.exit(1)
  }
}
