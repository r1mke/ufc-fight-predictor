"use client";

import { motion } from "framer-motion";
import type { MethodPrediction } from "@/types";

const LABELS: Record<keyof MethodPrediction, string> = {
  KO_TKO: "KO / TKO",
  Submission: "Submission",
  Decision: "Decision",
};

export function MethodProbabilityBars({ method }: { method: MethodPrediction }) {
  const entries = (Object.keys(LABELS) as (keyof MethodPrediction)[])
    .map((key) => ({ key, label: LABELS[key], value: method[key] }))
    .sort((a, b) => b.value - a.value);

  const top = entries[0]?.key;

  return (
    <div className="space-y-3">
      {entries.map((entry, i) => {
        const pct = (entry.value * 100).toFixed(1);
        const isTop = entry.key === top;
        return (
          <div key={entry.key} className="flex items-center gap-3">
            <div
              className={`w-24 shrink-0 text-sm ${isTop ? "font-semibold text-primary" : "text-muted-foreground"}`}
            >
              {entry.label}
            </div>
            <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-muted">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${entry.value * 100}%` }}
                transition={{ duration: 0.5, delay: i * 0.06 }}
                className={`h-full rounded-full ${isTop ? "bg-primary" : "bg-foreground/30"}`}
              />
            </div>
            <div
              className={`w-12 shrink-0 text-right text-sm ${isTop ? "font-semibold text-primary" : "text-muted-foreground"}`}
            >
              {pct}%
            </div>
          </div>
        );
      })}
    </div>
  );
}
