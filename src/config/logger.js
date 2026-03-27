import winston from 'winston'

const { combine, timestamp, colorize, printf, json } = winston.format

const devFormat = printf(({ level, message, timestamp, ...meta }) => {
  const metaStr = Object.keys(meta).length ? ` ${JSON.stringify(meta)}` : ''
  return `${timestamp} [${level}]: ${message}${metaStr}`
})

export const logger = winston.createLogger({
  level: process.env.NODE_ENV === 'production' ? 'info' : 'debug',
  format: combine(timestamp({ format: 'HH:mm:ss' })),
  transports: [
    new winston.transports.Console({
      format: combine(
        colorize(),
        timestamp({ format: 'HH:mm:ss' }),
        devFormat
      ),
    }),
    new winston.transports.File({
      filename: 'logs/error.log',
      level: 'error',
      format: json(),
    }),
    new winston.transports.File({
      filename: 'logs/combined.log',
      format: json(),
    }),
  ],
})
