import { scoreDevice, SCORE_THRESHOLD } from './deviceScoring.js'
import { isDeviceInDisconnectGrace } from './deviceDisconnectGrace.js'

const CONNECTED_STATUSES = new Set(['ACTIVE', 'TRAINING', 'IDLE'])

/**
 * Eligible devices for training, including DISCONNECTED agents still inside the grace window.
 */
export async function filterEligibleDevicesAsync(devices) {
  const out = []

  for (const device of devices) {
    if (device.status === 'DROPPED') continue

    if (device.status === 'DISCONNECTED') {
      if (!(await isDeviceInDisconnectGrace(device.id))) continue
    } else if (!CONNECTED_STATUSES.has(device.status)) {
      continue
    }

    const computeScore = scoreDevice(device)
    if (computeScore < SCORE_THRESHOLD) continue

    out.push({ ...device, computeScore })
  }

  return out.sort((a, b) => b.computeScore - a.computeScore)
}
