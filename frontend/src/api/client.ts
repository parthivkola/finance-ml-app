import axios from "axios";

// Point to the FastAPI backend
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "/api/v1",
  headers: {
    "Content-Type": "application/json",
  },
});

export interface ModelMetrics {
  id: number;
  model_name: string;
  version: string;
  trained_at: string;
  accuracy: number | null;
  train_accuracy: number | null;
  overfit_status: string | null;
  f1_score: number | null;
  roc_auc: number | null;
}

export interface BestModelResponse {
  model_name: string;
  version: string;
  accuracy: number | null;
  train_accuracy: number | null;
  overfit_status: string | null;
  f1_score: number | null;
  roc_auc: number | null;
  is_fallback: boolean;
}

export interface HealthResponse {
  status: string;
  version: string;
  models_available: string[];
}

export interface HistoryRecord {
  symbol: string;
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface NewsRecord {
  id: number;
  symbol: string;
  title: string;
  published_date: string;
  sentiment_score: number | null;
  sentiment_label: string | null;
}

export interface FeatureImpact {
  feature: string;
  impact: number;
}

export interface PredictResponse {
  symbol: string;
  model_name: string;
  prediction: "UP" | "DOWN";
  confidence: number;
  explanation: FeatureImpact[];
  indicators?: {
    sma_20: number | null;
    sma_50: number | null;
    rsi: number | null;
    macd: number | null;
  };
  disclaimer: string;
}

export const api = {
  getHealth: () => apiClient.get<HealthResponse>("/health").then(res => res.data),
  
  getModels: () => apiClient.get<ModelMetrics[]>("/models").then(res => res.data),

  getBestModel: () => apiClient.get<BestModelResponse>("/models/best").then(res => res.data),
  
  getHistory: (symbol: string, limit: number = 90) => 
    apiClient.get<HistoryRecord[]>(`/history/${symbol}?limit=${limit}`).then(res => res.data),
    
  getNews: (symbol: string, limit: number = 20) => 
    apiClient.get<NewsRecord[]>(`/news/${symbol}?limit=${limit}`).then(res => res.data),
    
  predict: (symbol: string, modelName: string) => 
    apiClient.post<PredictResponse>("/predict", { symbol, model_name: modelName }).then(res => res.data),
};

