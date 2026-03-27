// Prisma seed script
// Run with: npm run db:seed

import { PrismaClient } from '@prisma/client'
import { hash } from 'bcryptjs'

const prisma = new PrismaClient()

async function main() {
  // Create demo user
  const hashedPassword = await hash('password123', 10)

  const user = await prisma.user.upsert({
    where: { email: 'demo@idlefl.com' },
    update: {},
    create: {
      email: 'demo@idlefl.com',
      passwordHash: hashedPassword,
      userId: 'demo_user',
    },
  })

  console.log('Seeded demo user:', user.email)
}

main()
  .catch((e) => {
    console.error(e)
    process.exit(1)
  })
  .finally(async () => {
    await prisma.$disconnect()
  })