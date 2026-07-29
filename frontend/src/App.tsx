import { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { ModelSelector } from './components/ModelSelector';
import { PredictionPanel } from './components/PredictionPanel';
import { StockChart } from './components/StockChart';
import { ShapExplainer } from './components/ShapExplainer';
import { NewsFeed } from './components/NewsFeed';
import { api, type HistoryRecord, type NewsRecord, type PredictResponse } from './api/client';

function App() {
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedModelOverfitStatus, setSelectedModelOverfitStatus] = useState<string | null>(null);
  const [currentSymbol, setCurrentSymbol] = useState<string | null>(null);

  const [historyData, setHistoryData] = useState<HistoryRecord[]>([]);
  const [newsData, setNewsData] = useState<NewsRecord[]>([]);
  const [predictData, setPredictData] = useState<PredictResponse | null>(null);

  const handleModelSelect = (modelName: string, overfitStatus?: string | null) => {
    setSelectedModel(modelName);
    setSelectedModelOverfitStatus(overfitStatus ?? null);
  };

  const handlePredictionComplete = async (symbol: string, prediction: PredictResponse | null) => {
    setCurrentSymbol(symbol);
    setPredictData(prediction);

    if (prediction) {
      try {
        const [history, news] = await Promise.all([
          api.getHistory(symbol),
          api.getNews(symbol),
        ]);
        setHistoryData(history);
        setNewsData(news);
      } catch (err) {
        console.error('Error fetching auxiliary data:', err);
      }
    } else {
      setHistoryData([]);
      setNewsData([]);
    }
  };

  return (
    <div className="app-layout">
      <Sidebar />

      <main className="main-content">
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '1rem' }}>
          <div>
            <h1 style={{ fontSize: '2rem', marginBottom: '0.25rem' }}>Financial Market Intelligence</h1>
            <p style={{ margin: 0 }}>Technical analysis · NLP sentiment scoring · Multi-horizon forecasting</p>
          </div>
        </header>

        <ModelSelector
          selectedModel={selectedModel}
          onSelect={handleModelSelect}
        />

        <div className="dashboard-grid">
          {/* Left Column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <PredictionPanel
              selectedModel={selectedModel}
              selectedModelOverfitStatus={selectedModelOverfitStatus}
              onPredictionComplete={handlePredictionComplete}
            />
            {currentSymbol && historyData.length > 0 && (
              <StockChart data={historyData} />
            )}
          </div>

          {/* Right Column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            {predictData && predictData.explanation && (
              <ShapExplainer data={predictData.explanation} />
            )}
            {currentSymbol && (
              <NewsFeed data={newsData} />
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
