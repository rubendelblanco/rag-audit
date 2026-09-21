interface StatCardProps {
  label: string;
  value: string;
  variant?: "good" | "warn" | "bad" | "neutral";
}

export function StatCard({ label, value, variant = "neutral" }: StatCardProps) {
  return (
    <div className={`stat-card stat-${variant}`}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}
