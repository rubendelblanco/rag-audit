import { scoreLevel } from "../score";

interface ScoreBadgeProps {
  label: string;
  value: number;
}

export function ScoreBadge({ label, value }: ScoreBadgeProps) {
  return (
    <span className={`badge badge-${scoreLevel(value)}`}>
      {label} {value.toFixed(2)}
    </span>
  );
}
