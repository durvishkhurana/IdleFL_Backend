import fs from 'fs/promises'
import path from 'path'
import { fileURLToPath } from 'url'
import { asyncHandler } from '../../utils/asyncHandler.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export const downloadScript = asyncHandler(async (req, res) => {
  const os = req.query.os || 'windows'
  const serverUrl = process.env.CLIENT_URL?.replace('5173', '4000') || 'http://localhost:4000'

  const scriptName = os === 'mac' ? 'agent_mac.py' : 'agent_windows.py'
  const scriptPath = path.join(__dirname, '../../../scripts', scriptName)

  let scriptContent
  try {
    scriptContent = await fs.readFile(scriptPath, 'utf-8')
  } catch {
    return res.status(404).json({ error: 'Script not found' })
  }

  // Inject the correct server URL into the script
  scriptContent = scriptContent.replace('SERVER_URL_PLACEHOLDER', serverUrl)

  res.setHeader('Content-Type', 'text/plain')
  res.setHeader('Content-Disposition', `attachment; filename="${scriptName}"`)
  res.send(scriptContent)
})