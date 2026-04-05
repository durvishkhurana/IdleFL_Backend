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
 * Rebuilds the payload for a specific shard range. This is used for fault
 * recovery so a reassigned device receives the exact same slice of data.
 *
 * @param {Object} opts
 * @param {string | null | undefined} opts.datasetPath
 * @param {string | null | undefined} opts.datasetContent
 * @param {string} opts.modelType
 * @param {number} opts.shardStart
 * @param {number} opts.shardEnd
 * @returns {Promise<Object>}
 */
export async function getShardPayload({ datasetPath, datasetContent, modelType, shardStart, shardEnd }) {
  if (modelType === 'CNN') {
    const datasetName = resolveCnnDatasetName(datasetPath)
    const indices = Array.from({ length: Math.max(0, shardEnd - shardStart + 1) }, (_, offset) => shardStart + offset)
    return { datasetName, indices, format: 'images' }
  }

  if (!datasetContent?.trim()) {
    throw new Error('Tabular shard rebuild requires datasetContent on the training job')
  }

  const { rows } = loadTabularDatasetFromString(datasetContent, 'CSV dataset')
  const shardRows = rows.slice(shardStart, shardEnd + 1)
  return {
    X: shardRows.map((row) => row.X),
    y: shardRows.map((row) => row.y),
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
 * @param {string | null | undefined} opts.datasetContent
 * @param {Array} opts.devices
 * @param {string} opts.modelType
 * @returns {Promise<Array<Object>>}
 */
export async function partitionDataset({ datasetPath, datasetContent, devices, modelType }) {
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

  if (!datasetContent?.trim()) {
    throw new Error(`A CSV dataset is required for ${modelType}`)
  }

  const { rows, totalRows } = loadTabularDatasetFromString(datasetContent, 'CSV dataset')
  const selectedDevices = totalRows < devices.length ? devices.slice(0, totalRows) : devices
  const shardSizes = allocateShardSizes(totalRows, selectedDevices)

  let shardCursor = 0
  return selectedDevices.map((device, index) => {
    const shardSize = shardSizes[index]
    const shardStart = shardCursor
    const shardEnd = shardCursor + shardSize - 1
    const shardRows = rows.slice(shardStart, shardCursor + shardSize)
    shardCursor += shardSize

    return {
      deviceId: device.id,
      shardStart,
      shardEnd,
      shardSize,
      X: shardRows.map((row) => row.X),
      y: shardRows.map((row) => row.y),
    }
  })
}
