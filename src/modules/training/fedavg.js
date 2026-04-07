import { redis } from '../../config/redis.js'

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
/**
 * Float32Array requires byteOffset % 4 === 0 on the underlying ArrayBuffer.
 * Express `express.raw()` and some Redis clients return Buffer slices that are not aligned.
 */
function float32ArrayFromBuffer(buf) {
  if (!Buffer.isBuffer(buf)) {
    buf = Buffer.from(buf)
  }
  if (buf.byteLength % 4 !== 0) {
    throw new Error(`Invalid buffer length ${buf.byteLength} (not multiple of 4)`)
  }
  const copy = Buffer.from(buf)
  return new Float32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4)
}

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

export async function fedAvgAggregate(weightContributions) {
  // weightContributions: Array of { weights?: number[], weightsBinB64?: string, weightsBlobKey?: string, weightsLen?: number, shardSize: number, deviceId: string }
  if (!weightContributions || weightContributions.length === 0) {
    throw new Error('No weight contributions to aggregate')
  }

  const totalSamples = weightContributions.reduce((sum, c) => sum + c.shardSize, 0)

  const decodeWeights = async (contrib) => {
    if (Array.isArray(contrib.weights)) {
      return contrib.weights
    }
    if (typeof contrib.weightsBlobKey === 'string' && contrib.weightsBlobKey.length > 0) {
      let buf =
        typeof redis.getBuffer === 'function'
          ? await redis.getBuffer(contrib.weightsBlobKey)
          : await redis.get(contrib.weightsBlobKey)
      if (buf && !Buffer.isBuffer(buf)) {
        buf = Buffer.from(buf)
      }
      if (!buf || buf.byteLength % 4 !== 0) {
        throw new Error(`Invalid Redis vector for device ${contrib.deviceId}`)
      }
      const view = float32ArrayFromBuffer(buf)
      if (Number.isInteger(contrib.weightsLen) && contrib.weightsLen > 0 && contrib.weightsLen !== view.length) {
        throw new Error(
          `Weight length mismatch: expected ${contrib.weightsLen}, got ${view.length} from device ${contrib.deviceId}`
        )
      }
      return view
    }
    if (typeof contrib.weightsBinB64 === 'string' && contrib.weightsBinB64.length > 0) {
      const buf = Buffer.from(contrib.weightsBinB64, 'base64')
      if (buf.byteLength % 4 !== 0) {
        throw new Error(`Invalid binary weights byteLength=${buf.byteLength} from device ${contrib.deviceId}`)
      }
      const view = float32ArrayFromBuffer(buf)
      if (Number.isInteger(contrib.weightsLen) && contrib.weightsLen > 0 && contrib.weightsLen !== view.length) {
        throw new Error(
          `Weight length mismatch: expected ${contrib.weightsLen}, got ${view.length} from device ${contrib.deviceId}`
        )
      }
      return view
    }
    throw new Error(`Missing weights for device ${contrib.deviceId}`)
  }

  const first = await decodeWeights(weightContributions[0])
  const weightLength = first.length

  // Validate all weight vectors are the same length
  for (const contrib of weightContributions) {
    const w = await decodeWeights(contrib)
    if (w.length !== weightLength) {
      throw new Error(
        `Weight length mismatch: expected ${weightLength}, got ${w.length} from device ${contrib.deviceId}`
      )
    }
  }

  // Weighted average
  // Use Float64 accumulation for numerical stability; convert to JS array at the end for JSON transport.
  const acc = new Float64Array(weightLength)

  for (const contrib of weightContributions) {
    const proportion = contrib.shardSize / totalSamples
    const w = await decodeWeights(contrib)
    for (let i = 0; i < weightLength; i++) {
      acc[i] += proportion * w[i]
    }
  }

  return {
    globalWeights: Array.from(acc),
    totalSamples,
    numDevices: weightContributions.length,
  }
}