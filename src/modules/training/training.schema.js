import { z } from 'zod'

export const createJobSchema = z.object({
  body: z.object({
    sessionId: z.string().cuid('Invalid session ID'),
    modelType: z.enum(['LINEAR_REGRESSION', 'LOGISTIC_REGRESSION', 'CNN']),
    learningRate: z.number().min(0.0001).max(1).optional(),
    numRounds: z.number().int().min(1).max(100).optional(),
    batchSize: z.number().int().min(1).max(1024).optional(),
  }),
})

/** Multipart /start: body fields arrive as strings */
export const createJobMultipartSchema = z.object({
  body: z.object({
    sessionId: z.string().cuid('Invalid session ID'),
    modelType: z.enum(['LINEAR_REGRESSION', 'LOGISTIC_REGRESSION', 'CNN']),
    learningRate: z.coerce.number().min(0.0001).max(1).optional(),
    numRounds: z.coerce.number().int().min(1).max(100).optional(),
    batchSize: z.coerce.number().int().min(1).max(1024).optional(),
  }),
})

export const uploadDatasetSchema = z.object({
  body: z.object({
    sessionId: z.string().cuid('Invalid session ID'),
  }),
})
