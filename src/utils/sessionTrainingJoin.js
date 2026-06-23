import { prisma } from '../config/database.js'
import { logger } from '../config/logger.js'
import { DISCONNECT_GRACE_SECONDS, isDeviceInDisconnectGrace } from './deviceDisconnectGrace.js'

/** Max seconds after disconnect to rejoin an in-progress round (defaults to disconnect grace). */
export const TRAINING_REJOIN_SECONDS = parseInt(
  process.env.TRAINING_REJOIN_SECONDS || String(DISCONNECT_GRACE_SECONDS),
  10
)

/**
 * Active RUNNING job for a session, if any.
 */
export async function getActiveTrainingJob(sessionId) {
  return prisma.trainingJob.findFirst({
    where: { sessionId, status: 'RUNNING' },
    orderBy: { startedAt: 'desc' },
  })
}

/**
 * Whether this device may (re)join the current round of an active job.
 * @returns {Promise<{ allowed: boolean, reason?: string, job?: object, assignment?: object }>}
 */
export async function evaluateTrainingRejoin(deviceId, sessionId, { wasDisconnectGrace = false } = {}) {
  const job = await getActiveTrainingJob(sessionId)
  if (!job) {
    return { allowed: true, job: null }
  }

  const assignment = await prisma.taskAssignment.findFirst({
    where: {
      jobId: job.id,
      deviceId,
      roundNum: job.currentRound,
      status: 'IN_PROGRESS',
    },
    include: { job: true },
  })

  if (!assignment) {
    return {
      allowed: false,
      reason: 'not_assigned_current_round',
      job,
    }
  }

  const inGrace = wasDisconnectGrace || (await isDeviceInDisconnectGrace(deviceId))
  const device = await prisma.device.findUnique({
    where: { id: deviceId },
    select: { status: true, updatedAt: true },
  })

  if (device?.status === 'DROPPED') {
    return { allowed: false, reason: 'device_dropped', job, assignment }
  }

  if (!inGrace && device?.status === 'DISCONNECTED') {
    const msSinceUpdate = Date.now() - new Date(device.updatedAt).getTime()
    if (msSinceUpdate > TRAINING_REJOIN_SECONDS * 1000) {
      return { allowed: false, reason: 'rejoin_window_expired', job, assignment }
    }
  }

  return { allowed: true, job, assignment }
}

/**
 * Devices that already participated in this job (any round). New session devices are excluded from later rounds.
 */
export async function filterToJobParticipantDevices(jobId, devices) {
  if (!devices?.length) return []

  const rows = await prisma.taskAssignment.findMany({
    where: { jobId },
    select: { deviceId: true },
    distinct: ['deviceId'],
  })
  const participantIds = new Set(rows.map((r) => r.deviceId))
  return devices.filter((d) => participantIds.has(d.id))
}

/**
 * @param {import('socket.io').Socket} socket
 */
export function emitTrainingJoinBlocked(socket, { reason, job, message }) {
  const payload = {
    reason,
    jobId: job?.id,
    currentRound: job?.currentRound,
    message:
      message ||
      (reason === 'training_in_progress_new_device'
        ? 'Training is already running. New devices cannot join until this job finishes.'
        : 'You are not part of the current training round.'),
  }
  socket.emit('training:session_locked', payload)
  logger.info(`Training join blocked for device ${socket.deviceId}: ${reason}`)
}
