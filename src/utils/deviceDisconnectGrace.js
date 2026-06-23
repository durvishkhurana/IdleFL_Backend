import { redis } from '../config/redis.js'
import { logger } from '../config/logger.js'

/** Seconds before a disconnected device is treated as dropped for task reassignment. */
export const DISCONNECT_GRACE_SECONDS = parseInt(process.env.DISCONNECT_GRACE_SECONDS || '15', 10)

const reassignTimers = new Map()

function graceRedisKey(deviceId) {
  return `device:${deviceId}:disconnect_grace`
}

export async function markDeviceDisconnected(deviceId) {
  const at = Date.now()
  await redis.setex(graceRedisKey(deviceId), DISCONNECT_GRACE_SECONDS + 60, String(at))
}

export async function clearDeviceDisconnectGrace(deviceId) {
  cancelScheduledReassign(deviceId)
  await redis.del(graceRedisKey(deviceId)).catch(() => {})
}

export async function isDeviceInDisconnectGrace(deviceId) {
  const raw = await redis.get(graceRedisKey(deviceId))
  if (!raw) return false
  const at = parseInt(raw, 10)
  if (!Number.isFinite(at)) return false
  return Date.now() - at < DISCONNECT_GRACE_SECONDS * 1000
}

export function scheduleReassignAfterGrace(deviceId, callback) {
  cancelScheduledReassign(deviceId)
  const timer = setTimeout(() => {
    reassignTimers.delete(deviceId)
    Promise.resolve(callback()).catch((e) => logger.error(`disconnect grace reassign error (${deviceId}):`, e))
  }, DISCONNECT_GRACE_SECONDS * 1000)
  reassignTimers.set(deviceId, timer)
  logger.info(
    `Device ${deviceId}: reassignment deferred ${DISCONNECT_GRACE_SECONDS}s (disconnect grace)`
  )
}

export function cancelScheduledReassign(deviceId) {
  const timer = reassignTimers.get(deviceId)
  if (timer) {
    clearTimeout(timer)
    reassignTimers.delete(deviceId)
  }
}
