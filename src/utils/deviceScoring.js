// Device scoring formula from design doc
// Score(d) = 0.25*(1-CPU%) + 0.25*freeRam + 0.25*reliability - 0.25*activeTasks
// All inputs normalized 0-1, output is 0-1 float
// Devices scoring below SCORE_THRESHOLD are excluded from assignment

export const SCORE_THRESHOLD = 0.3

export function scoreDevice({ cpuPercent, freeRamGb, totalRamGb, reliabilityScore, activeTasks, computeType }) {
  const w1 = 0.25, w2 = 0.25, w3 = 0.25, w4 = 0.25

  const cpuNorm = cpuPercent / 100
  const ramNorm = totalRamGb > 0 ? freeRamGb / totalRamGb : 0
  const activeTasksNorm = Math.min(activeTasks / 5, 1)

  let score = w1 * (1 - cpuNorm) + w2 * ramNorm + w3 * reliabilityScore - w4 * activeTasksNorm

  // GPU bonus: CUDA = +0.15, MPS = +0.10
  if (computeType === 'CUDA') score += 0.15
  if (computeType === 'MPS') score += 0.10

  return Math.max(0, Math.min(1, score))
}

export function filterEligibleDevices(devices) {
  return devices
    .map((d) => ({ ...d, computeScore: scoreDevice(d) }))
    .filter((d) => d.computeScore >= SCORE_THRESHOLD && d.status !== 'DROPPED' && d.status !== 'DISCONNECTED')
    .sort((a, b) => b.computeScore - a.computeScore)
}
