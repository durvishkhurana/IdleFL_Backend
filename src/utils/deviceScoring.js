// Device scoring formula from design doc
// Score(d) = 0.25*(1-CPU%) + 0.25*freeRam + 0.25*reliability - 0.25*activeTasks
// All inputs normalized 0-1, output is 0-1 float
// Devices scoring below SCORE_THRESHOLD are excluded from assignment

export const SCORE_THRESHOLD = 0.3

export function scoreDevice({
  cpuPercent,
  freeRamGb,
  totalRamGb,
  reliabilityScore,
  activeTasks,
  computeType,
  gpuVramTotal,
  gpuVramUsed,
}) {
  const w1 = 0.25, w2 = 0.25, w3 = 0.25, w4 = 0.25
  const w5 = 0.20, w6 = 0.10

  const cpuNorm = cpuPercent / 100
  const ramNorm = totalRamGb > 0 ? freeRamGb / totalRamGb : 0
  const activeTasksNorm = Math.min(activeTasks / 5, 1)

  let score = w1 * (1 - cpuNorm) + w2 * ramNorm + w3 * reliabilityScore - w4 * activeTasksNorm

  // GPU bonus: CUDA = +0.15, MPS = +0.10
  if (computeType === 'CUDA') score += 0.15
  if (computeType === 'MPS') score += 0.10

  // GPU weighting: higher score → larger shard assigned by dataPartitioner
  // → GPU device processes more data per round → round time dominated
  // by fast device not CPU stragglers. w5+w6 intentionally outweigh
  // CPU/RAM factors for CUDA devices to maximize hardware utilization.
  // Reference: v2 improvement over IEEE IATMSI-2026 paper (v1 lacked GPU weighting)
  const hasGPU = computeType === 'CUDA' || computeType === 'MPS' ? 1.0 : 0
  const freeVRAMRatioRaw =
    gpuVramTotal && gpuVramTotal > 0
      ? (Number(gpuVramTotal) - Number(gpuVramUsed || 0)) / Number(gpuVramTotal)
      : 0
  const freeVRAMRatio = Math.max(0, Math.min(1, freeVRAMRatioRaw))
  score += w5 * hasGPU + w6 * freeVRAMRatio

  return Math.max(0, Math.min(1, score))
}

export function filterEligibleDevices(devices) {
  return devices
    .map((d) => ({ ...d, computeScore: scoreDevice(d) }))
    .filter((d) => d.computeScore >= SCORE_THRESHOLD && d.status !== 'DROPPED' && d.status !== 'DISCONNECTED')
    .sort((a, b) => b.computeScore - a.computeScore)
}
