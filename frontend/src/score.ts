export type ScoreLevel = "good" | "warn" | "bad";

export function scoreLevel(value: number): ScoreLevel {
  if (value >= 0.8) return "good";
  if (value >= 0.5) return "warn";
  return "bad";
}
