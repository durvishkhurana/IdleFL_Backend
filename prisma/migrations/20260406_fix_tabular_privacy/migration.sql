-- AlterTable: remove full CSV storage; keep tabular metadata for partitioning only.
ALTER TABLE "training_jobs" DROP COLUMN IF EXISTS "datasetContent";

ALTER TABLE "training_jobs" ADD COLUMN IF NOT EXISTS "totalRows" INTEGER;
ALTER TABLE "training_jobs" ADD COLUMN IF NOT EXISTS "numFeatures" INTEGER;
ALTER TABLE "training_jobs" ADD COLUMN IF NOT EXISTS "columnNames" TEXT;
ALTER TABLE "training_jobs" ADD COLUMN IF NOT EXISTS "datasetHash" TEXT;
