import { SessionService } from './session.service.js'
import { asyncHandler } from '../../utils/asyncHandler.js'

const sessionService = new SessionService()

export const createSession = asyncHandler(async (req, res) => {
  const session = await sessionService.createSession(req.user.id)
  res.status(201).json({ session })
})

export const joinSession = asyncHandler(async (req, res) => {
  const { sessionCode } = req.body
  const session = await sessionService.joinSession(sessionCode, req.user.id)
  res.json({ session })
})

export const getSession = asyncHandler(async (req, res) => {
  const session = await sessionService.getSession(req.params.id, req.user.id)
  res.json({ session })
})

export const getMySessions = asyncHandler(async (req, res) => {
  const sessions = await sessionService.getUserSessions(req.user.id)
  res.json({ sessions })
})
