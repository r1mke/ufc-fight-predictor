import { Trophy } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { FighterCard } from "@/components/FighterCard";
import { MethodProbabilityBars } from "@/components/MethodProbabilityBars";
import { ModelMetricsPanel } from "@/components/ModelMetricsPanel";
import { describeFeature } from "@/lib/glossary";
import type { PredictResponse } from "@/types";

export function PredictionResult({ result }: { result: PredictResponse }) {
  const loser = result.winner.fighter_id === result.fighter1.id ? result.fighter2 : result.fighter1;

  return (
    <div className="space-y-6">
      <Card className="border-primary/40">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <Trophy className="size-5 text-primary" />
            Predicted winner: {result.winner.fighter_name}
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            {(result.winner.probability * 100).toFixed(1)}% probability over {loser.name}
          </p>
        </CardHeader>
        <CardContent className="space-y-6">
          <div>
            <p className="mb-3 text-sm font-medium">Method of victory</p>
            <MethodProbabilityBars method={result.method} />
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <ModelMetricsPanel title="Winner model" metrics={result.winner_model_metrics} />
            <ModelMetricsPanel title="Method model" metrics={result.method_model_metrics} />
          </div>

          <div>
            <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Most influential factors (method of victory)
            </p>
            <div className="flex flex-wrap gap-2">
              {result.top_features.slice(0, 6).map((f) => {
                const label = f.feature.replace(/_diff$/, "").replace(/_/g, " ");
                const description = describeFeature(f.feature);
                const pill = (
                  <span className="cursor-help rounded-full border px-2.5 py-1 text-xs text-muted-foreground">
                    {label}
                  </span>
                );
                return description ? (
                  <Tooltip key={f.feature}>
                    <TooltipTrigger asChild>{pill}</TooltipTrigger>
                    <TooltipContent className="max-w-64">{description}</TooltipContent>
                  </Tooltip>
                ) : (
                  <span key={f.feature}>{pill}</span>
                );
              })}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FighterCard fighter={result.fighter1} />
        <FighterCard fighter={result.fighter2} />
      </div>
    </div>
  );
}
