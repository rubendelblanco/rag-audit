import { useEffect, useState } from "react";
import { fetchReport, type AuditDataset } from "./api";
import { StatCard } from "./components/StatCard";
import { ItemCard } from "./components/ItemCard";
import { scoreLevel } from "./score";

export default function App() {
  const [dataset, setDataset] = useState<AuditDataset | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchReport()
      .then(setDataset)
      .catch((err: Error) => setError(err.message));
  }, []);

  if (error) {
    return (
      <div className="app-shell">
        <div className="empty-state">
          <h1>RAG-Audit</h1>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (!dataset) {
    return (
      <div className="app-shell">
        <div className="empty-state">
          <p>Loading report…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>RAG-Audit</h1>
        <div className="stats-row">
          <StatCard label="Queries" value={String(dataset.total_queries)} />
          <StatCard
            label="Avg Faithfulness"
            value={dataset.avg_faithfulness.toFixed(2)}
            variant={scoreLevel(dataset.avg_faithfulness)}
          />
          <StatCard
            label="Avg Relevance"
            value={dataset.avg_relevance.toFixed(2)}
            variant={scoreLevel(dataset.avg_relevance)}
          />
        </div>
      </header>

      <main className="item-list">
        {dataset.items.map((item) => (
          <ItemCard key={item.id} item={item} />
        ))}
      </main>
    </div>
  );
}
