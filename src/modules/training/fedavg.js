/**
 * FedAvg Aggregation — McMahan et al. 2017
 *
 * W_global = Σ_k ( (n_k / N) * W_k )
 *
 * W_k   = weight tensor from device k (flat array of numbers)
 * n_k   = shard size for device k
 * N     = total dataset size
 *
 * Note: We use SGD (not Adam) because Adam tracks per-parameter momentum state.
 * Across independent workers that state is inconsistent and cannot be averaged.
 * SGD weights aggregate cleanly with this formula.
 */

/**
 * Shard-size–weighted mean of per-device loss and accuracy.
 *
 * @param {Array<{ shardSize?: number, loss?: number, accuracy?: number }>} contributions
 * @returns {{ avgLoss: number, avgAccuracy: number }}
 */
export function computeWeightedRoundMetrics(contributions) {
  const totalSamples = contributions.reduce((sum, c) => sum + (c.shardSize || 0), 0)
  if (totalSamples === 0) {
    return { avgLoss: 0, avgAccuracy: 0 }
  }
  const avgLoss = contributions.reduce((sum, c) => sum + (Number(c.loss) || 0) * (c.shardSize || 0), 0) / totalSamples
  const avgAccuracy =
    contributions.reduce((sum, c) => sum + (Number(c.accuracy) || 0) * (c.shardSize || 0), 0) / totalSamples
  return { avgLoss, avgAccuracy }
}

export function fedAvgAggregate(weightContributions) {
  // weightContributions: Array of { weights: number[], shardSize: number, deviceId: string }
  if (!weightContributions || weightContributions.length === 0) {
    throw new Error('No weight contributions to aggregate')
  }

  const totalSamples = weightContributions.reduce((sum, c) => sum + c.shardSize, 0)
  const weightLength = weightContributions[0].weights.length

  // Validate all weight vectors are the same length
  for (const contrib of weightContributions) {
    if (contrib.weights.length !== weightLength) {
      throw new Error(
        `Weight length mismatch: expected ${weightLength}, got ${contrib.weights.length} from device ${contrib.deviceId}`
      )
    }
  }

  // Weighted average
  const globalWeights = new Array(weightLength).fill(0)

  for (const contrib of weightContributions) {
    const proportion = contrib.shardSize / totalSamples
    for (let i = 0; i < weightLength; i++) {
      globalWeights[i] += proportion * contrib.weights[i]
    }
  }

  return {
    globalWeights,
    totalSamples,
    numDevices: weightContributions.length,
  }
}