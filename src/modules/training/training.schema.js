import { z } from 'zod'

export const createJobSchema = z.object({
  body: z.object({
    sessionId: z.string().cuid('Invalid session ID'),
    modelType: z.enum(['LINEAR_REGRESSION', 'LOGISTIC_REGRESSION', 'CNN']),
    learningRate: z.coerce.number().min(0.0001).max(1).optional(),
    numRounds: z.coerce.number().int().min(1).max(100).optional(),
    batchSize: z.coerce.number().int().min(1).max(1024).optional(),
    mu: z.coerce.number().min(0).max(1).optional(),

    // Tabular metadata (CSV is never uploaded; extracted in browser).
    totalRows: z.coerce.number().int().min(1).optional().nullable(),
    numFeatures: z.coerce.number().int().min(1).optional().nullable(),
    columnNames: z.string().optional().nullable(),
    datasetHash: z.string().optional().nullable(),

    // CNN hint (frontend sends 'mnist.zip' to select MNIST behavior).
    datasetPath: z.string().optional().nullable(),
  }),
})

export const uploadDatasetSchema = z.object({
  body: z.object({
    sessionId: z.string().cuid('Invalid session ID'),
  }),
})
