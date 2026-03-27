import { AuthService } from './auth.service.js'
import { asyncHandler } from '../../utils/asyncHandler.js'

const authService = new AuthService()

export const register = asyncHandler(async (req, res) => {
  const result = await authService.register(req.body)
  res.status(201).json(result)
})

export const login = asyncHandler(async (req, res) => {
  const result = await authService.login(req.body)
  res.json(result)
})

export const getMe = asyncHandler(async (req, res) => {
  const user = await authService.getMe(req.user.id)
  res.json({ user })
})
