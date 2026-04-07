import Redis from 'ioredis'
import { logger } from './logger.js'

export const redis = new Redis(process.env.REDIS_URL || 'redis://localhost:6379', {
  retryDelayOnFailover: 100,
  maxRetriesPerRequest: 3,
  lazyConnect: true,
})

redis.on('connect', () => logger.info('✓ Redis connected'))
redis.on('error', (err) => logger.error('✗ Redis error:', err))
redis.on('close', () => logger.warn('Redis connection closed'))

// Key helpers — all Redis keys go through here for consistency
export const REDIS_KEYS = {
  device: (deviceId) => `device:${deviceId}`,
  deviceHeartbeat: (deviceId) => `heartbeat:${deviceId}`,
  sessionDevices: (sessionId) => `session:${sessionId}:devices`,
  sessionInfo: (sessionId) => `session:${sessionId}:info`,
  jobState: (jobId) => `job:${jobId}:state`,
  jobRound: (jobId) => `job:${jobId}:round`,
  jobWeights: (jobId, round) => `job:${jobId}:round:${round}:weights`,
  /** Raw Float32 weight vector (Buffer) — avoids huge base64 in JSON metadata */
  jobWeightsVector: (jobId, round, deviceId) => `job:${jobId}:round:${round}:vec:${deviceId}`,
  jobWeightsChunk: (jobId, round, deviceId) => `job:${jobId}:round:${round}:weights_chunk:${deviceId}`,
  jobEval: (jobId, round) => `job:${jobId}:round:${round}:eval`,
  jobGlobalWeights: (jobId) => `job:${jobId}:global_weights`,
  jobRoundFinalized: (jobId, round) => `job:${jobId}:round:${round}:finalized`,
  jobRoundAggLock: (jobId, round) => `job:${jobId}:round:${round}:agg_lock`,
  deviceTask: (deviceId) => `device:${deviceId}:task`,
}

export async function connectRedis() {
  try {
    await redis.connect()
    logger.info('✓ Redis ready')
  } catch (error) {
    logger.error('✗ Redis connection failed:', error)
    process.exit(1)
  }
}
