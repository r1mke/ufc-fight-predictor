import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";
import { MODEL_LABELS, type ModelMetrics } from "@/types";

function Metric({ label, value, tooltip }: { label: string; value: string; tooltip?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center gap-1 text-[11px] uppercase tracking-wide text-muted-foreground">
        {label}
        {tooltip && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="size-3 cursor-help" />
            </TooltipTrigger>
            <TooltipContent className="max-w-64">{tooltip}</TooltipContent>
          </Tooltip>
        )}
      </div>
      <span className="text-sm font-medium">{value}</span>
    </div>
  );
}

export function ModelMetricsPanel({ title, metrics }: { title: string; metrics: ModelMetrics }) {
  return (
    <div className="rounded-lg border p-3">
      <p className="mb-2 text-xs font-medium text-muted-foreground">
        {title} · {MODEL_LABELS[metrics.model_name]}
      </p>
      <div className="grid grid-cols-4 gap-3">
        <Metric label="Accuracy" value={`${(metrics.accuracy * 100).toFixed(1)}%`} />
        <Metric label="Macro F1" value={metrics.macro_f1.toFixed(3)} />
        <Metric
          label="Log-loss"
          value={metrics.log_loss.toFixed(3)}
          tooltip="Measures how well-calibrated the predicted probabilities are - lower means the confidence percentages shown can be trusted more. Not just raw accuracy."
        />
        <Metric
          label="CV accuracy"
          value={`${(metrics.cv_mean * 100).toFixed(1)}% ± ${(metrics.cv_std * 100).toFixed(1)}`}
          tooltip="Mean ± standard deviation across 5-fold cross-validation - shows how stable this result is, not just a single train/test split."
        />
      </div>
    </div>
  );
}
