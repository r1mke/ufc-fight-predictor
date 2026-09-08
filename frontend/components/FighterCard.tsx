import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { FighterDetail } from "@/types";

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</span>
      <span className="text-sm font-medium">{value}</span>
    </div>
  );
}

function pct(value: number | null): string {
  return value == null ? "—" : `${(value * 100).toFixed(0)}%`;
}

function num(value: number | null, digits = 1): string {
  return value == null ? "—" : value.toFixed(digits);
}

export function FighterCard({ fighter }: { fighter: FighterDetail }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-2">
          <div>
            <CardTitle className="text-lg">{fighter.name}</CardTitle>
            <p className="text-sm text-muted-foreground">
              {fighter.wins}-{fighter.losses}-{fighter.draws} · {fighter.weight_class}
            </p>
          </div>
          {fighter.win_streak > 0 && (
            <Badge className="bg-primary/90">{fighter.win_streak}-fight win streak</Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-3">
          <Stat label="Height" value={fighter.height_in ? `${fighter.height_in}"` : "—"} />
          <Stat label="Reach" value={fighter.reach_in ? `${fighter.reach_in.toFixed(0)}"` : "—"} />
          <Stat label="Age" value={fighter.age_years ? `${Math.round(fighter.age_years)}` : "—"} />
          <Stat label="Stance" value={fighter.stance} />
          <Stat label="Weight" value={fighter.weight_lbs ? `${fighter.weight_lbs} lbs` : "—"} />
          <Stat label="UFC fights" value={String(fighter.fight_count)} />
        </div>
        <div className="border-t pt-3">
          <p className="mb-2 text-[11px] uppercase tracking-wide text-muted-foreground">
            Career stats
          </p>
          <div className="grid grid-cols-4 gap-3">
            <Stat label="Str/min" value={num(fighter.career_stats.slpm)} />
            <Stat label="Str Acc" value={pct(fighter.career_stats.str_acc)} />
            <Stat label="TD Avg" value={num(fighter.career_stats.td_avg)} />
            <Stat label="Sub Avg" value={num(fighter.career_stats.sub_avg)} />
          </div>
        </div>
        {(fighter.reach_missing || fighter.height_missing) && (
          <p className="text-[11px] text-muted-foreground">
            Some physical measurements were estimated (not on record).
          </p>
        )}
      </CardContent>
    </Card>
  );
}
