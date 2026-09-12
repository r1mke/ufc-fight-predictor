"use client";

import { useState } from "react";
import { CompareForm } from "@/components/CompareForm";
import { Glossary } from "@/components/Glossary";
import { PredictionResult } from "@/components/PredictionResult";
import { predictFight } from "@/lib/api";
import type { FighterSummary, ModelName, ModelVariant, PredictResponse } from "@/types";

export default function Home() {
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handlePredict(params: {
    fighter1: FighterSummary;
    fighter2: FighterSummary;
    weightClass: string;
    modelName: ModelName;
    variant: ModelVariant;
  }) {
    setLoading(true);
    setError(null);
    try {
      const response = await predictFight({
        fighter1_id: params.fighter1.id,
        fighter2_id: params.fighter2.id,
        weight_class: params.weightClass,
        model_name: params.modelName,
        variant: params.variant,
      });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto w-full max-w-4xl flex-1 px-4 py-10 sm:px-6">
      <header className="mb-8 space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">UFC Fight Predictor</h1>
        <p className="text-sm text-muted-foreground">
          Pick two fighters and a weight class, choose a model, and get a data-driven prediction
          of the winner and method of victory.
        </p>
      </header>

      <CompareForm onPredict={handlePredict} loading={loading} />

      {error && (
        <p className="mt-6 rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </p>
      )}

      {result && (
        <div className="mt-8">
          <PredictionResult result={result} />
        </div>
      )}

      <div className="mt-8">
        <Glossary />
      </div>
    </main>
  );
}
