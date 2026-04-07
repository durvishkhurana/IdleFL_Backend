import path from 'path'

const CNN_DATASET_SIZES = {
  mnist: 60_000,
  cifar10: 50_000,
}

/**
 * Parses a single CSV line while handling quoted cells.
 *
 * @param {string} line
 * @returns {string[]}
 */
function parseCsvLine(line) {
  const cells = []
  let current = ''
  let inQuotes = false

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index]
    const nextChar = line[index + 1]

    if (char === '"') {
      if (inQuotes && nextChar === '"') {
        current += '"'
        index += 1
      } else {
        inQuotes = !inQuotes
      }
      continue
    }

    if (char === ',' && !inQuotes) {
      cells.push(current.trim())
      current = ''
      continue
    }

    current += char
  }

  if (inQuotes) {
    throw new Error('Unterminated quoted CSV field')
  }

  cells.push(current.trim())
  return cells
}

/**
 * Converts raw CSV text into numeric features and targets.
 *
 * @param {string} rawCsv
 * @param {string} label - used in error messages
 * @returns {{ rows: Array<{ X: number[], y: number }>, totalRows: number }}
 */
function loadTabularDatasetFromString(rawCsv, label = 'CSV dataset') {
  const lines = String(rawCsv)
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)

  if (lines.length < 2) {
    throw new Error(`Malformed ${label}: expected a header row and at least one data row`)
  }

  const header = parseCsvLine(lines[0])
  if (header.length < 2) {
    throw new Error(`Malformed ${label}: expected at least one feature column and one target column`)
  }

  const rows = lines.slice(1).map((line, lineIndex) => {
    const cells = parseCsvLine(line)

    if (cells.length !== header.length) {
      throw new Error(
        `Malformed ${label} on data row ${lineIndex + 2}: expected ${header.length} columns, received ${cells.length}`
      )
    }

    const numericValues = cells.map((cell) => Number(cell))
    const invalidIndex = numericValues.findIndex((value) => Number.isNaN(value))

    if (invalidIndex !== -1) {
      throw new Error(
        `Malformed ${label} on data row ${lineIndex + 2}: column "${header[invalidIndex]}" is not numeric`
      )
    }

    return {
      X: numericValues.slice(0, -1),
      y: numericValues[numericValues.length - 1],
    }
  })

  return { rows, totalRows: rows.length }
}

/**
 * Parses uploaded CSV once at job creation: row count, feature count, header JSON — no row payload stored server-side.
 *
 * @param {string} rawCsv
 * @returns {{ totalRows: number, numFeatures: number, columnNames: string }}
 */
export function extractTabularTrainingMetadata(rawCsv) {
  const lines = String(rawCsv)
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)

  if (lines.length < 2) {
    throw new Error('Malformed CSV dataset: expected a header row and at least one data row')
  }

  const header = parseCsvLine(lines[0])
  if (header.length < 2) {
    throw new Error('Malformed CSV dataset: expected at least one feature column and one target column')
  }

  const numFeatures = header.length - 1
  let totalRows = 0

  for (let i = 1; i < lines.length; i += 1) {
    const cells = parseCsvLine(lines[i])
    if (cells.length !== header.length) {
      throw new Error(
        `Malformed CSV dataset on data row ${i + 1}: expected ${header.length} columns, received ${cells.length}`
      )
    }
    const numericValues = cells.map((cell) => Number(cell))
    const invalidIndex = numericValues.findIndex((value) => Number.isNaN(value))
    if (invalidIndex !== -1) {
      throw new Error(
        `Malformed CSV dataset on data row ${i + 1}: column "${header[invalidIndex]}" is not numeric`
      )
    }
    totalRows += 1
  }

  return {
    totalRows,
    numFeatures,
    columnNames: JSON.stringify(header),
  }
}

/**
 * Rebuilds the payload for a specific shard range. Used for fault recovery / reassignment.
 * Tabular shards are indices only — agents load rows locally from DATASET_PATH.
 *
 * @param {Object} opts
 * @param {string | null | undefined} opts.datasetPath
 * @param {string} opts.modelType
 * @param {number} opts.shardStart
 * @param {number} opts.shardEnd
 * @param {number} opts.shardSize
 * @returns {Promise<Object>}
 */
export async function getShardPayload({ datasetPath, modelType, shardStart, shardEnd, shardSize }) {
  if (modelType === 'CNN') {
    const datasetName = resolveCnnDatasetName(datasetPath)
    const indices = Array.from({ length: Math.max(0, shardEnd - shardStart + 1) }, (_, offset) => shardStart + offset)
    return { datasetName, indices, format: 'images' }
  }

  return {
    shardStart,
    shardEnd,
    shardSize,
    format: 'tabular',
  }
}

/**
 * Allocates shard sizes while keeping at least one sample per selected device.
 *
 * @param {number} totalSamples
 * @param {Array<{ computeScore: number }>} devices
 * @returns {number[]}
 */
function allocateShardSizes(totalSamples, devices) {
  if (devices.length === 0) {
    throw new Error('No eligible devices for partitioning')
  }

  if (devices.length === 1) {
    return [totalSamples]
  }

  const selectedDevices = totalSamples < devices.length ? devices.slice(0, totalSamples) : devices
  const baseSizes = new Array(selectedDevices.length).fill(totalSamples >= selectedDevices.length ? 1 : 0)
  let remaining = totalSamples - baseSizes.reduce((sum, size) => sum + size, 0)

  if (remaining <= 0) {
    return baseSizes
  }

  const totalScore = selectedDevices.reduce((sum, device) => sum + Math.max(device.computeScore || 0, 0), 0) || selectedDevices.length
  const proportionalExtras = selectedDevices.map((device) => Math.floor((Math.max(device.computeScore || 0, 0) / totalScore) * remaining))

  let used = 0
  for (let index = 0; index < selectedDevices.length; index += 1) {
    baseSizes[index] += proportionalExtras[index]
    used += proportionalExtras[index]
  }

  let remainder = remaining - used
  let cursor = 0
  while (remainder > 0) {
    baseSizes[cursor] += 1
    cursor = (cursor + 1) % selectedDevices.length
    remainder -= 1
  }

  return baseSizes
}

/**
 * Resolves the built-in image dataset name for CNN jobs.
 *
 * @param {string | null | undefined} datasetPath
 * @returns {'mnist' | 'cifar10'}
 */
function resolveCnnDatasetName(datasetPath) {
  const normalized = path.basename(datasetPath || '').toLowerCase()
  return normalized.includes('cifar') ? 'cifar10' : 'mnist'
}

/**
 * Partitions a dataset into shards for federated distribution.
 * Stronger devices receive proportionally larger shards.
 *
 * @param {Object} opts
 * @param {string | null | undefined} opts.datasetPath
 * @param {number | null | undefined} opts.totalRows — required for tabular models (LINEAR / LOGISTIC)
 * @param {Array} opts.devices
 * @param {string} opts.modelType
 * @returns {Promise<Array<Object>>}
 */
export async function partitionDataset({ datasetPath, totalRows, devices, modelType }) {
  if (!Array.isArray(devices) || devices.length === 0) {
    throw new Error('No eligible devices for partitioning')
  }

  if (modelType === 'CNN') {
    if (!datasetPath) {
      throw new Error('A dataset upload filename is required for CNN (e.g. mnist.zip)')
    }
    const datasetName = resolveCnnDatasetName(datasetPath)
    const totalSamples = CNN_DATASET_SIZES[datasetName]
    const selectedDevices = totalSamples < devices.length ? devices.slice(0, totalSamples) : devices
    const shardSizes = allocateShardSizes(totalSamples, selectedDevices)

    let shardCursor = 0
    return selectedDevices.map((device, index) => {
      const shardSize = shardSizes[index]
      const shardStart = shardCursor
      const shardEnd = shardCursor + shardSize - 1
      const indices = Array.from({ length: shardSize }, (_, offset) => shardStart + offset)
      shardCursor += shardSize

      return {
        deviceId: device.id,
        shardStart,
        shardEnd,
        shardSize,
        datasetName,
        indices,
      }
    })
  }

  const tabularTotal = totalRows
  if (tabularTotal === null || tabularTotal === undefined || Number.isNaN(Number(tabularTotal)) || tabularTotal < 1) {
    throw new Error(`A valid totalRows is required for ${modelType}`)
  }

  const totalSamples = Number(tabularTotal)
  const selectedDevices = totalSamples < devices.length ? devices.slice(0, totalSamples) : devices
  const shardSizes = allocateShardSizes(totalSamples, selectedDevices)

  let shardCursor = 0
  return selectedDevices.map((device, index) => {
    const shardSize = shardSizes[index]
    const shardStart = shardCursor
    const shardEnd = shardCursor + shardSize - 1
    shardCursor += shardSize

    return {
      deviceId: device.id,
      shardStart,
      shardEnd,
      shardSize,
    }
  })
}

export { loadTabularDatasetFromString, parseCsvLine }
