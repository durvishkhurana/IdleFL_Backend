import { logger } from '../../config/logger.js'

// Whitelisted commands — no shell injection possible
const ALLOWED_COMMANDS = [
  'nvidia-smi', 'python --version', 'python3 --version',
  'pip list', 'pip3 list', 'ls', 'dir', 'pwd',
  'cat /proc/cpuinfo', 'cat /proc/meminfo',
  'df -h', 'free -h', 'top -bn1',
  'echo', 'uname -a', 'whoami', 'date',
  'torch --version', 'python -c "import torch; print(torch.__version__)"',
  'python -c "import torch; print(torch.cuda.is_available())"',
]

function isCommandAllowed(command) {
  const trimmed = command.trim().toLowerCase()
  return ALLOWED_COMMANDS.some((allowed) => trimmed.startsWith(allowed.toLowerCase()))
}

export function registerTerminalHandlers(io, socket) {

  // Coordinator sends a command to a specific device
  socket.on('terminal:input', ({ targetDeviceId, data }) => {
    try {
      if (!data || !targetDeviceId) return

      if (!isCommandAllowed(data)) {
        socket.emit('terminal:output', {
          deviceId: targetDeviceId,
          data: `\r\n\x1b[31m[IdleFL Security] Command not allowed: "${data.trim()}"\x1b[0m\r\nAllowed: ${ALLOWED_COMMANDS.join(', ')}\r\n`,
        })
        logger.warn(`Blocked terminal command from ${socket.user?.email}: "${data.trim()}"`)
        return
      }

      logger.debug(`Terminal: routing command to device ${targetDeviceId}: ${data.trim()}`)

      io.sockets.sockets.forEach((s) => {
        if (s.deviceId === targetDeviceId) {
          s.emit('terminal:execute', { command: data.trim(), fromSocketId: socket.id })
        }
      })
    } catch (error) {
      logger.error('terminal:input error:', error)
    }
  })

  // Agent emits output back from command execution
  socket.on('terminal:output', ({ data, targetSocketId }) => {
    try {
      const targetSocket = io.sockets.sockets.get(targetSocketId)
      if (targetSocket) {
        targetSocket.emit('terminal:output', { deviceId: socket.deviceId, data })
      }
    } catch (error) {
      logger.error('terminal:output error:', error)
    }
  })
}