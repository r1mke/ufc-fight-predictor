export interface FighterSummary {
  id: string;
  name: string;
  weight_class: string;
  wins: number;
  losses: number;
  draws: number;
  height_in: number | null;
  reach_in: number | null;
}

export interface FighterCareerStats {
  slpm: number | null;
  str_acc: number | null;
  sapm: number | null;
  str_def: number | null;
  td_avg: number | null;
  td_acc: number | null;
  td_def: number | null;
  sub_avg: number | null;
}

export interface FighterDetail extends FighterSummary {
  stance: string;
  age_years: number | null;
  weight_lbs: number | null;
  reach_missing: boolean;
  height_missing: boolean;
  career_stats: FighterCareerStats;
  win_streak: number;
  fight_count: number;
}

export type ModelName = "logistic_regression" | "random_forest" | "lightgbm";

export const MODEL_LABELS: Record<ModelName, string> = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  lightgbm: "LightGBM",
};

export interface ModelMetrics {
  model_name: ModelName;
  accuracy: number;
  macro_f1: number;
  log_loss: number;
  cv_mean: number;
  cv_std: number;
  test_size: number;
}

export interface FeatureImportance {
  feature: string;
  score: number;
}

export interface TargetMetrics {
  labels: string[];
  models: ModelMetrics[];
  anova_feature_importance: FeatureImportance[];
}

export interface CorrelationPair {
  feature_a: string;
  feature_b: string;
  r: number;
}

export interface ModelsResponse {
  winner: TargetMetrics;
  method: TargetMetrics;
  correlation_pairs: CorrelationPair[];
}

export interface PredictRequest {
  fighter1_id: string;
  fighter2_id: string;
  weight_class: string;
  model_name: ModelName;
  is_title_fight?: boolean;
}

export interface WinnerPrediction {
  fighter_id: string;
  fighter_name: string;
  probability: number;
}

export interface MethodPrediction {
  KO_TKO: number;
  Submission: number;
  Decision: number;
}

export interface PredictResponse {
  winner: WinnerPrediction;
  method: MethodPrediction;
  model_name: ModelName;
  winner_model_metrics: ModelMetrics;
  method_model_metrics: ModelMetrics;
  top_features: FeatureImportance[];
  fighter1: FighterDetail;
  fighter2: FighterDetail;
}
