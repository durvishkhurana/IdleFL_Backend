import { z } from 'zod'

export const registerSchema = z.object({
  body: z.object({
    email: z.string().email('Invalid email'),
    password: z.string().min(8, 'Password must be at least 8 characters'),
  }),
})

export const loginSchema = z.object({
  body: z.object({
    email: z.string().email('Invalid email'),
    password: z.string().min(1, 'Password required'),
  }),
})

export const agentLoginSchema = z.object({
  body: z.object({
    agentId: z.string().min(1, 'agentId is required'),
    sessionCode: z.string().regex(/^FL-\d+$/, 'sessionCode must be in FL-NNNN format'),
  }),
})
