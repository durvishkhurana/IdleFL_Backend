/** Local epochs per federated round (Socket config sent to agents). */
export const EPOCHS_PER_MODEL = {
  LINEAR_REGRESSION: 10,
  LOGISTIC_REGRESSION: 5,
  CNN: 1,
}

export function epochsForModelType(modelType) {
  return EPOCHS_PER_MODEL[modelType] ?? 1
}
