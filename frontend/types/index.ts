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

// Feature-engineering variant: how a fighter matchup is turned into model
// input. "diff" (default, production) subtracts fighter B's stats from
// fighter A's; "concat" (experimental) keeps both fighters' stats as separate
// columns and lets the model learn its own comparison.
export type ModelVariant = "diff" | "concat";

export const MODEL_VARIANT_LABELS: Record<ModelVariant, string> = {
  diff: "Standard (A − B difference)",
  concat: "Experimental (side-by-side stats)",
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
  variant?: ModelVariant;
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
  variant: ModelVariant;
  winner_model_metrics: ModelMetrics;
  method_model_metrics: ModelMetrics;
  top_features: FeatureImportance[];
  fighter1: FighterDetail;
  fighter2: FighterDetail;
}

// --- Data acquisition (scraping / manual submission / review) -------------

export interface NewFighterInfo {
  name: string;
  height_in: number | null;
  reach_in: number | null;
  weight_lbs: number | null;
  stance: string | null;
  dob: string | null;
}

export type FightMethod = "KO_TKO" | "Submission" | "Decision";

export interface GranularStats {
  f1_sig_landed: number | null;
  f1_sig_att: number | null;
  f2_sig_landed: number | null;
  f2_sig_att: number | null;
  f1_td_landed: number | null;
  f1_td_att: number | null;
  f2_td_landed: number | null;
  f2_td_att: number | null;
  f1_ctrl_sec: number | null;
  f2_ctrl_sec: number | null;
  f1_kd: number | null;
  f2_kd: number | null;
  f1_sub_att: number | null;
  f2_sub_att: number | null;
}

export interface PendingFightCreate extends GranularStats {
  fighter1_id: string | null;
  fighter2_id: string | null;
  fighter1_name: string;
  fighter2_name: string;
  weight_class: string;
  method: FightMethod;
  winner_name: string;
  event_date: string;
  is_title_fight: boolean;
  new_fighter_1: NewFighterInfo | null;
  new_fighter_2: NewFighterInfo | null;
  source_url: string | null;
}

export type PendingFightUpdate = Partial<PendingFightCreate>;

export interface PendingFight extends PendingFightCreate {
  id: string;
  source: "manual" | "scraped";
  status: "pending" | "approved" | "rejected";
  submitted_at: string;
  reviewed_at: string | null;
}

export interface JobStatus {
  status: "idle" | "running" | "done" | "error";
  started_at: string | null;
  finished_at: string | null;
  error_message: string | null;
  result_summary: string | null;
}

export type ScrapeStatus = JobStatus;
export type RetrainStatus = JobStatus;

export interface ModelVersion {
  version_id: string;
  created_at: string;
  is_current: boolean;
  metrics_summary: Record<string, Record<string, { accuracy: number; log_loss: number }>>;
}

export interface ModelVersionsResponse {
  versions: ModelVersion[];
}
