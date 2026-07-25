import React from 'react';
import { type NewsRecord } from '../api/client';
import { ExternalLink, Clock } from 'lucide-react';
import clsx from 'clsx';

interface Props {
  data: NewsRecord[];
}

export const NewsFeed: React.FC<Props> = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <div className="glass-panel" style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
        No news articles found for this symbol.
      </div>
    );
  }

  const getBadgeClass = (label: string | null) => {
    if (label === 'positive') return 'badge-positive';
    if (label === 'negative') return 'badge-negative';
    return 'badge-neutral';
  };

  return (
    <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%', maxHeight: '600px' }}>
      <h3 style={{ marginBottom: '1rem', color: 'var(--text-secondary)', fontSize: '1rem' }}>
        Market Intelligence Feed
      </h3>
      <div style={{ overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '1rem', paddingRight: '0.5rem' }}>
        {data.map(article => (
          <div key={article.id} style={{ padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem', marginBottom: '0.75rem' }}>
              <h4 style={{ fontSize: '0.9rem', lineHeight: 1.4, margin: 0, color: 'var(--text-primary)' }}>
                {article.title}
              </h4>
              <a href={`https://www.google.com/search?q=${encodeURIComponent(article.title)}`} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--text-muted)', flexShrink: 0 }}>
                <ExternalLink size={16} />
              </a>
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                <Clock size={14} />
                <span>{new Date(article.published_date).toLocaleDateString()}</span>
              </div>
              
              <div className={clsx("badge", getBadgeClass(article.sentiment_label))}>
                {article.sentiment_label || 'Unscored'}
                {article.sentiment_score !== null && ` (${article.sentiment_score > 0 ? '+' : ''}${article.sentiment_score.toFixed(2)})`}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
