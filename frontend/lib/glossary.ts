export interface GlossaryEntry {
  term: string;
  description: string;
}

export const FIGHTER_STAT_GLOSSARY: GlossaryEntry[] = [
  { term: "Height / Reach", description: "Fighter's height and arm span (wingspan) in inches. A longer reach lets a fighter strike from further away." },
  { term: "Stance", description: "Fighting stance: Orthodox (left hand/foot back), Southpaw (right hand/foot back), Switch (alternates), or Unknown if not on record." },
  { term: "UFC fights", description: "Number of fights this fighter has in the UFC fight history used by the model (career-to-date). Can be lower than the official total record if some bouts had no clear winner (draws, no-contests, disqualifications)." },
  { term: "Str/min (SLpM)", description: "Significant Strikes Landed per Minute — average striking volume across the fighter's career." },
  { term: "Str Acc", description: "Significant Strike Accuracy — percentage of significant strikes that land out of all attempts." },
  { term: "TD Avg", description: "Average number of takedowns landed per 15 minutes of fight time." },
  { term: "Sub Avg", description: "Average number of submission attempts per 15 minutes of fight time." },
];

export const MODEL_METRIC_GLOSSARY: GlossaryEntry[] = [
  { term: "Accuracy", description: "Percentage of test fights the model predicted correctly. Simple but can be misleading with imbalanced classes (e.g. always guessing the most common outcome looks decent)." },
  { term: "Macro F1", description: "Average F1-score treating every outcome class equally, regardless of how common it is. A model that ignores rare outcomes (like Submission) scores worse here even if its raw accuracy looks fine." },
  { term: "Log-loss", description: "Measures how well-calibrated the predicted probabilities are, not just whether the top guess was right. Lower is better — it means when the model says '80% confident', it's actually right about 80% of the time." },
  { term: "CV accuracy", description: "Mean accuracy (± standard deviation) across 5 different train/test splits (cross-validation). Shows how stable the result is, rather than relying on a single lucky or unlucky split." },
];

interface FeatureInfo {
  term: string;
  description: string;
}

// Keyed by the raw feature name with any _a / _b / _diff suffix stripped.
export const FEATURE_GLOSSARY: Record<string, FeatureInfo> = {
  height_in: { term: "Height", description: "Difference in height between the two fighters." },
  reach_in: { term: "Reach", description: "Difference in arm reach — a big edge here favors the fighter who can strike from distance." },
  weight_lbs: { term: "Weight", description: "Difference in fighting weight." },
  age_years: { term: "Age", description: "Difference in age at the time of the fight." },
  prior_fight_count: { term: "UFC experience", description: "Difference in UFC experience (number of prior fights)." },
  prior_wins: { term: "Prior wins", description: "Difference in career UFC wins going into the fight." },
  prior_losses: { term: "Prior losses", description: "Difference in career UFC losses going into the fight." },
  win_streak: { term: "Win streak", description: "Difference in current win streak — momentum coming into the fight." },
  days_since_last_fight: { term: "Days since last fight", description: "Difference in days since each fighter's last bout (ring rust vs. freshness)." },
  prior_avg_sig_landed: { term: "Significant strikes landed", description: "Difference in average significant strikes landed per fight, based on career history." },
  prior_avg_sig_att: { term: "Significant strikes attempted", description: "Difference in average significant strikes attempted per fight." },
  prior_avg_td_landed: { term: "Takedowns landed", description: "Difference in average takedowns landed per fight." },
  prior_avg_td_att: { term: "Takedowns attempted", description: "Difference in average takedown attempts per fight." },
  prior_avg_sub_att: { term: "Submission attempts", description: "Difference in average submission attempts per fight — higher favors a Submission finish." },
  prior_avg_ctrl_sec: { term: "Control time", description: "Difference in average ground/clinch control time (seconds) per fight." },
  prior_avg_kd: { term: "Knockdowns", description: "Difference in average knockdowns scored per fight — higher favors a KO/TKO finish." },
};

export function describeFeature(rawFeatureName: string): string | null {
  const key = rawFeatureName.replace(/_diff$/, "").replace(/_[ab]$/, "");
  return FEATURE_GLOSSARY[key]?.description ?? null;
}
