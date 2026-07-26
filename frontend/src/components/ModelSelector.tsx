import React, { useEffect, useState } from 'react';
import { api, type ModelMetrics } from '../api/client';

interface Props {
  selectedModel: string;
  onSelect: (modelName: string) => void;
}

export const ModelSelector: React.FC<Props> = ({ selectedModel, onSelect }) => {
  const [models, setModels] = useState<ModelMetrics[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getModels()
      .then(data => {
        // Only keep valid multi-horizon models (e.g. xgboost_1d, lightgbm_3d)
        const VALID = /^(xgboost|lightgbm|random_forest|logistic_regression)_(1|3|5)d$/;
        const filtered = data.filter(m => VALID.test(m.model_name));
        // Dedup by model_name, taking highest version
        const unique = filtered.reduce((acc, curr) => {
          if (!acc[curr.model_name] || acc[curr.model_name].version < curr.version) {
            acc[curr.model_name] = curr;
          }
          return acc;
        }, {} as Record<string, ModelMetrics>);
        setModels(Object.values(unique));
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div style={{ color: 'var(--text-secondary)' }}>Loading models...</div>;
  }

  const selectedData = models.find(m => m.model_name === selectedModel);

  return (
    <div className="glass-panel model-selector-container">
      <div>
        <label style={{ display: 'block', fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
          Select Champion Model
        </label>
        <select 
          className="input-field"
          value={selectedModel}
          onChange={(e) => onSelect(e.target.value)}
          style={{ width: '250px', textTransform: 'capitalize' }}
        >
          {models.map(m => (
            <option key={m.model_name} value={m.model_name}>
              {m.model_name.replace(/_/g, ' ')}
            </option>
          ))}
        </select>
      </div>

      {selectedData && (
        <div className="model-stats-border" style={{ display: 'flex', gap: '1.5rem' }}>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ACCURACY</div>
            <div style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              {(selectedData.accuracy! * 100).toFixed(1)}%
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>F1 SCORE</div>
            <div style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              {selectedData.f1_score?.toFixed(3)}
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ROC-AUC</div>
            <div style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              {selectedData.roc_auc?.toFixed(3)}
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>STATUS</div>
            <div style={{ 
              fontSize: '1rem', 
              fontWeight: 600, 
              color: selectedData.overfit_status?.includes('OVERFIT') ? 'var(--accent-down)' : 'var(--accent-up)',
              display: 'flex',
              alignItems: 'center',
              height: '1.25rem'
            }}>
              {selectedData.overfit_status || '✅ OK'}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
