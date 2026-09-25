import { Trophy } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FighterCard } from "@/components/FighterCard";
import { MethodProbabilityBars } from "@/components/MethodProbabilityBars";
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
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FighterCard fighter={result.fighter1} />
        <FighterCard fighter={result.fighter2} />
      </div>
    </div>
  );
}
