"use client";

import { useEffect, useState } from "react";
import { Loader2, Swords } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { FighterSearch } from "@/components/FighterSearch";
import { WEIGHT_CLASSES } from "@/lib/constants";
import {
  MODEL_LABELS,
  MODEL_VARIANT_LABELS,
  type FighterSummary,
  type ModelName,
  type ModelVariant,
} from "@/types";

interface Props {
  onPredict: (params: {
    fighter1: FighterSummary;
    fighter2: FighterSummary;
    weightClass: string;
    modelName: ModelName;
    variant: ModelVariant;
  }) => void;
  loading: boolean;
}

const MODEL_NAMES: ModelName[] = ["logistic_regression", "random_forest", "lightgbm"];
const MODEL_VARIANTS: ModelVariant[] = ["diff", "concat"];

export function CompareForm({ onPredict, loading }: Props) {
  const [fighter1, setFighter1] = useState<FighterSummary | null>(null);
  const [fighter2, setFighter2] = useState<FighterSummary | null>(null);
  const [weightClass, setWeightClass] = useState<string>("");
  const [weightClassTouched, setWeightClassTouched] = useState(false);
  const [modelName, setModelName] = useState<ModelName>("lightgbm");
  const [variant, setVariant] = useState<ModelVariant>("diff");

  // Auto-suggest the weight class from whichever fighter was picked, unless
  // the user has already overridden it themselves.
  useEffect(() => {
    if (weightClassTouched) return;
    const suggestion = [fighter1?.weight_class, fighter2?.weight_class].find((wc) =>
      wc ? WEIGHT_CLASSES.includes(wc) : false
    );
    if (suggestion) {
      setWeightClass(suggestion);
    }
  }, [fighter1, fighter2, weightClassTouched]);

  const canSubmit = fighter1 && fighter2 && weightClass && fighter1.id !== fighter2.id && !loading;

  return (
    <div className="space-y-5 rounded-xl border bg-card p-5">
      <div className="grid grid-cols-1 items-end gap-4 md:grid-cols-[1fr_auto_1fr]">
        <FighterSearch label="Fighter 1" selected={fighter1} onSelect={setFighter1} exclude={fighter2?.id} />
        <div className="hidden justify-center pb-2 md:flex">
          <Swords className="size-5 text-muted-foreground" />
        </div>
        <FighterSearch label="Fighter 2" selected={fighter2} onSelect={setFighter2} exclude={fighter1?.id} />
      </div>

      {fighter1 && fighter2 && fighter1.id === fighter2.id && (
        <p className="text-sm text-destructive">Pick two different fighters.</p>
      )}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium text-muted-foreground">Weight class</label>
          <Select
            value={weightClass}
            onValueChange={(v) => {
              setWeightClass(v);
              setWeightClassTouched(true);
            }}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select category" />
            </SelectTrigger>
            <SelectContent>
              {WEIGHT_CLASSES.map((wc) => (
                <SelectItem key={wc} value={wc}>
                  {wc}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium text-muted-foreground">Model</label>
          <Select value={modelName} onValueChange={(v) => setModelName(v as ModelName)}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {MODEL_NAMES.map((m) => (
                <SelectItem key={m} value={m}>
                  {MODEL_LABELS[m]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium text-muted-foreground">Feature engineering</label>
          <Select value={variant} onValueChange={(v) => setVariant(v as ModelVariant)}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {MODEL_VARIANTS.map((v) => (
                <SelectItem key={v} value={v}>
                  {MODEL_VARIANT_LABELS[v]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <Button
        className="w-full"
        size="lg"
        disabled={!canSubmit}
        onClick={() =>
          fighter1 &&
          fighter2 &&
          onPredict({ fighter1, fighter2, weightClass, modelName, variant })
        }
      >
        {loading && <Loader2 className="mr-2 size-4 animate-spin" />}
        Predict
      </Button>
    </div>
  );
}
