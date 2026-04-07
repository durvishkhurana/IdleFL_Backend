import 'dotenv/config'
import http from 'http'
import { createApp } from './src/app.js'
import { createSocketServer } from './src/socket/socket.manager.js'
import { connectDatabase } from './src/config/database.js'
import { connectRedis } from './src/config/redis.js'
import { startHeartbeatMonitor } from './src/jobs/heartbeat.monitor.js'
import { TrainingService } from './src/modules/training/training.service.js'
import { initTrainingController } from './src/modules/training/training.controller.js'
import { logger } from './src/config/logger.js'
import { prisma } from './src/config/database.js'

const PORT = process.env.PORT || 4000

async function bootstrap() {
  await connectDatabase()
  await connectRedis()

  const app = createApp()
  const httpServer = http.createServer(app)
  const io = createSocketServer(httpServer)

  const trainingService = new TrainingService(io)
  initTrainingController(trainingService)

  startHeartbeatMonitor(io)

  setInterval(async () => {
    const now = new Date();
    const result = await prisma.session.updateMany({
      where: {
        status: { in: ['WAITING', 'ACTIVE'] },
        expiresAt: { lt: now }
      },
      data: { status: 'EXPIRED' }
    });
    logger.info(`Session expiry job: flipped ${result.count} sessions to EXPIRED`);
  }, 5 * 60 * 1000);

  httpServer.listen(PORT, () => {
    logger.info(``)
    logger.info(`  ██ ██████  ██      ███████ ███████ ██`)
    logger.info(`  ██ ██   ██ ██      ██      ██      ██`)
    logger.info(`  ██ ██   ██ ██      █████   █████   ██`)
    logger.info(`  ██ ██   ██ ██      ██      ██      ██`)
    logger.info(`  ██ ██████  ███████ ███████ ██      ███████`)
    logger.info(``)
    logger.info(`  IdleFL Backend v1.0`)
    logger.info(`  Distributed Federated Learning Platform`)
    logger.info(``)
    logger.info(`  ✓ HTTP   → http://localhost:${PORT}`)
    logger.info(`  ✓ WS     → ws://localhost:${PORT}`)
    logger.info(`  ✓ Health → http://localhost:${PORT}/health`)
    logger.info(`  ✓ Mode   → ${process.env.NODE_ENV || 'development'}`)
    logger.info(``)
  })

  process.on('SIGTERM', async () => {
    logger.info('SIGTERM received — graceful shutdown')
    httpServer.close(async () => {
      process.exit(0)
    })
  })
}

bootstrap().catch((error) => {
  console.error('Failed to start server:', error)
  process.exit(1)
})