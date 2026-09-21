export interface AtomicClaim {
  statement: string;
  supported: boolean;
  reason: string;
}

export interface EvaluationMetrics {
  faithfulness: number;
  answer_relevance: number;
  reasoning: string | null;
  is_abstention: boolean;
  claims: AtomicClaim[];
}

export interface RAGItem {
  id: string;
  question: string;
  contexts: string[];
  answer: string;
  ground_truth: string | null;
  metrics: EvaluationMetrics | null;
}

export interface AuditDataset {
  total_queries: number;
  avg_faithfulness: number;
  avg_relevance: number;
  items: RAGItem[];
}

export async function fetchReport(): Promise<AuditDataset> {
  const res = await fetch("/api/report");
  if (!res.ok) {
    if (res.status === 404) {
      throw new Error("No dataset loaded. Run `rag-audit <file>` to evaluate one.");
    }
    throw new Error(`Request failed with status ${res.status}`);
  }
  return res.json();
}
