import { useState } from "react";
import type { RAGItem } from "../api";
import { ScoreBadge } from "./ScoreBadge";

interface ItemCardProps {
  item: RAGItem;
}

export function ItemCard({ item }: ItemCardProps) {
  const [expanded, setExpanded] = useState(false);
  const metrics = item.metrics;

  return (
    <div className="item-card">
      <button className="item-header" onClick={() => setExpanded((v) => !v)}>
        <span className="item-question">{item.question}</span>
        <span className="item-badges">
          {metrics?.is_abstention && <span className="badge badge-neutral">Abstention</span>}
          {metrics && <ScoreBadge label="Faithfulness" value={metrics.faithfulness} />}
          {metrics && <ScoreBadge label="Relevance" value={metrics.answer_relevance} />}
          <span className="item-chevron">{expanded ? "▾" : "▸"}</span>
        </span>
      </button>

      {expanded && (
        <div className="item-body">
          <section>
            <h4>Contexts</h4>
            {item.contexts.map((context, i) => (
              <p key={i} className="context-block">
                {context}
              </p>
            ))}
          </section>

          <section>
            <h4>Answer</h4>
            <p>{item.answer}</p>
          </section>

          {item.ground_truth && (
            <section>
              <h4>Ground truth</h4>
              <p>{item.ground_truth}</p>
            </section>
          )}

          {metrics && metrics.reasoning && (
            <section>
              <h4>Relevance reasoning</h4>
              <p>{metrics.reasoning}</p>
            </section>
          )}

          {metrics && metrics.claims.length > 0 && (
            <section>
              <h4>Claims</h4>
              <ul className="claims-list">
                {metrics.claims.map((claim, i) => (
                  <li key={i} className={claim.supported ? "claim-supported" : "claim-unsupported"}>
                    <span className="claim-icon">{claim.supported ? "✓" : "✗"}</span>
                    <div>
                      <div className="claim-statement">{claim.statement}</div>
                      {claim.reason && <div className="claim-reason">{claim.reason}</div>}
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
