import { z } from 'zod'

export const joinSessionSchema = z.object({
  body: z.object({
    sessionCode: z.string().regex(/^FL-\d{4}$/, 'Invalid session code format. Expected FL-XXXX'),
  }),
})
