/** Local epochs per federated round (Socket config sent to agents). */
export const EPOCHS_PER_MODEL = {
  LINEAR_REGRESSION: 10,
  LOGISTIC_REGRESSION: 5,
  CNN: 1,
}

export const PARTICIPATION_THRESHOLD = 0.7
// Aggregate when this fraction of assigned devices have submitted.
// 0.7 = 70% threshold. Stragglers are skipped for this round only,
// not dropped from the session. mu=0 default keeps FedAvg behaviour.

export const ROUND_TIMEOUT_SECONDS = parseInt(process.env.ROUND_TIMEOUT_SECONDS) || 120
// Fallback: aggregate with whoever submitted after this many seconds
// even if PARTICIPATION_THRESHOLD was never reached.

/** Grace period before reassigning tasks after a socket disconnect (seconds). */
export const DISCONNECT_GRACE_SECONDS = parseInt(process.env.DISCONNECT_GRACE_SECONDS) || 15

export function epochsForModelType(modelType) {
  return EPOCHS_PER_MODEL[modelType] ?? 1
}
