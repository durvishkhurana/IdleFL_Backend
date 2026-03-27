// Generates human-readable session IDs like FL-4829
export function generateSessionCode() {
  const digits = Math.floor(1000 + Math.random() * 9000)
  return `FL-${digits}`
}

// Generates user-friendly user IDs like "user_a3k9"
export function generateUserId(email) {
  const prefix = email.split('@')[0].toLowerCase().replace(/[^a-z0-9]/g, '').slice(0, 10)
  const suffix = Math.random().toString(36).slice(2, 6)
  return `${prefix}_${suffix}`
}
