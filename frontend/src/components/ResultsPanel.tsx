import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { EyeOff, Activity, Cpu, CheckCircle2, List } from 'lucide-react';

interface Prediction {
  label: string;
  confidence: number;
  ranked: Array<{ class: string, probability: number }>;
  error?: string;
}

interface ResultsPanelProps {
  results: {
    cnn?: Prediction;
    rf?: Prediction;
  } | null;
  isClassifying: boolean;
  metrics: {
    rf_accuracy: number | null;
    cnn_accuracy: number | null;
  } | null;
  activeTab: 'preview' | 'classification';
  setActiveTab: (tab: 'preview' | 'classification') => void;
  previewUrl: string | null;
}

const ResultsPanel: React.FC<ResultsPanelProps> = ({ results, isClassifying, metrics, activeTab, setActiveTab, previewUrl }) => {

  return (
    <aside className="results-panel">
      <div className="section-header">
        <h3 className="technical-label">Results</h3>
        <div className="status-badge">
          {isClassifying ? 'Analyzing Input' : (results ? 'Complete' : 'Waiting for Input')}
        </div>
      </div>

      <div className="tabs-header">
        <button 
          className={`tab-btn ${activeTab === 'preview' ? 'active' : ''}`}
          onClick={() => setActiveTab('preview')}
        >
          <EyeOff size={14} />
          <span>Preview</span>
        </button>
        <button 
          className={`tab-btn ${activeTab === 'classification' ? 'active' : ''}`}
          onClick={() => setActiveTab('classification')}
        >
          <CheckCircle2 size={14} />
          <span>Classification</span>
        </button>
      </div>

      <div className="results-content-container">
        <AnimatePresence mode="wait">
          {activeTab === 'preview' ? (
            <motion.div 
              key="preview"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              className="tab-content"
            >
              <div className="inference-preview-container">
                {previewUrl ? (
                  <div className="preview-active">
                    <img src={previewUrl} alt="Inference Preview" className="preview-image" />
                    {isClassifying && <div className="scanning-line" />}
                    <div className="preview-overlay">
                       <div className="overlay-tag technical-label">Neural_Layer_01</div>
                    </div>
                  </div>
                ) : (
                  <div className="preview-empty">
                    <EyeOff size={32} color="var(--bg-surface-highest)" />
                    <span className="preview-text">INFERENCE_PREVIEW_NULL</span>
                  </div>
                )}
              </div>
            </motion.div>
          ) : (
            <motion.div 
              key="classification"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              className="tab-content"
            >
              <div className="classification-results">
                {results ? (
                  <div className="final-results">
                    {results.cnn && (
                      <div className="result-block">
                        <div className="block-title technical-label">CNN_RESULT</div>
                        <div className="label-badge">{results.cnn.label}</div>
                      </div>
                    )}
                    {results.rf && (
                      <div className="result-block">
                        <div className="block-title technical-label">RF_RESULT</div>
                        <div className="label-badge secondary">{results.rf.label}</div>
                      </div>
                    )}
                    
                    <div className="probability-table">
                      <div className="table-header technical-label">
                        <List size={12} />
                        <span>Probability Ranking</span>
                      </div>
                      {(results.cnn || results.rf)?.ranked.map((item: any, idx: number) => (
                        <div key={idx} className="table-row">
                          <span className="row-class">{item.class}</span>
                          <span className="row-prob">{(item.probability * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="preview-empty">
                    <CheckCircle2 size={32} color="var(--bg-surface-highest)" />
                    <span className="preview-text">NO_DATA_AVAILABLE</span>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="confidence-metrics">
        <div className="metric-item">
          <div className="metric-info">
            <span className="metric-label">CNN_CONFIDENCE</span>
            <span className="metric-value">{results?.cnn?.confidence ? `${(results.cnn.confidence * 100).toFixed(0)}%` : '0%'}</span>
          </div>
          <div className="progress-track">
            <motion.div 
              className="progress-bar cnn"
              initial={{ width: 0 }}
              animate={{ width: results?.cnn?.confidence ? `${results.cnn.confidence * 100}%` : 0 }}
            />
          </div>
        </div>

        <div className="metric-item">
          <div className="metric-info">
            <span className="metric-label">RF_CONFIDENCE</span>
            <span className="metric-value">{results?.rf?.confidence ? `${(results.rf.confidence * 100).toFixed(0)}%` : '0%'}</span>
          </div>
          <div className="progress-track">
            <motion.div 
              className="progress-bar rf"
              initial={{ width: 0 }}
              animate={{ width: results?.rf?.confidence ? `${results.rf.confidence * 100}%` : 0 }}
            />
          </div>
        </div>
      </div>

      <div className="explanation-text">
        <p>"Upload an image to trigger the inference engine and view neural layer activation maps."</p>
      </div>

      <div className="bottom-stats">
        <div className="stat-card">
          <div className="stat-header">
            <Activity size={14} color="var(--primary)" />
            <span className="stat-label">Avg Latency</span>
          </div>
          <div className="stat-value">-- ms</div>
        </div>
        <div className="stat-card">
          <div className="stat-header">
            <Cpu size={14} color="var(--success)" />
            <span className="stat-label">Compute Load</span>
          </div>
          <div className="stat-value active">
            <span className="pulse-dot"></span>
            Idle
          </div>
        </div>
      </div>

      <style jsx>{`
        .results-panel {
          background: var(--bg-deep);
          border-left: 1px solid var(--border-color);
          padding: var(--spacing-md);
          display: flex;
          flex-direction: column;
          gap: var(--spacing-md);
        }

        .section-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: var(--spacing-xs);
        }

        .status-badge {
          background: var(--bg-surface-high);
          padding: 4px 8px;
          border-radius: var(--radius-full);
          font-size: 10px;
          font-family: var(--font-mono);
          text-transform: uppercase;
          color: var(--text-muted);
          border: 1px solid var(--border-color);
        }

        .tabs-header {
          display: flex;
          background: var(--bg-surface-high);
          padding: 4px;
          border-radius: var(--radius-sm);
          gap: 4px;
        }

        .tab-btn {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 6px;
          padding: 8px;
          font-size: 12px;
          font-weight: 600;
          color: var(--text-muted);
          background: transparent;
          border-radius: 4px;
        }

        .tab-btn.active {
          background: var(--bg-surface-highest);
          color: var(--text-primary);
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }

        .results-content-container {
          min-height: 280px;
        }

        .inference-preview-container, .classification-results {
          aspect-ratio: 16/10;
          background: radial-gradient(circle at center, var(--bg-surface-highest) 0%, var(--bg-surface) 100%);
          border-radius: var(--radius-md);
          border: 1px solid var(--border-color);
          display: flex;
          align-items: center;
          justify-content: center;
          position: relative;
          overflow: hidden;
          width: 100%;
        }

        .preview-empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: var(--spacing-xs);
        }

        .preview-text {
          font-family: var(--font-mono);
          font-size: 11px;
          color: var(--bg-surface-highest);
          letter-spacing: 0.1em;
        }

        .preview-active {
           width: 100%;
           height: 100%;
           display: flex;
           align-items: center;
           justify-content: center;
           position: relative;
        }

        .preview-image {
          width: 100%;
          height: 100%;
          object-fit: cover;
          opacity: 0.6;
          filter: grayscale(0.5) contrast(1.2);
        }

        .preview-overlay {
          position: absolute;
          top: var(--spacing-sm);
          left: var(--spacing-sm);
        }

        .overlay-tag {
          font-size: 9px;
          background: rgba(99, 102, 241, 0.2);
          color: var(--primary);
          padding: 2px 6px;
          border-radius: 2px;
          border: 1px solid var(--primary);
        }

        .scanning-line {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 2px;
          background: var(--primary);
          box-shadow: 0 0 10px var(--primary);
          animation: scan 2s infinite ease-in-out;
          z-index: 10;
        }

        @keyframes scan {
          0% { top: 0; }
          50% { top: 100%; }
          100% { top: 0; }
        }
        
        .preview-label {
           font-family: var(--font-mono);
           font-size: 11px;
           color: var(--primary);
           letter-spacing: 0.1em;
        }

        .final-results {
          width: 100%;
          padding: var(--spacing-md);
          display: flex;
          flex-direction: column;
          gap: var(--spacing-md);
        }

        .result-block {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .block-title {
          font-size: 10px;
          color: var(--text-muted);
        }

        .label-badge {
          background: rgba(99, 102, 241, 0.1);
          border: 1px solid var(--primary);
          color: var(--primary);
          padding: 8px 12px;
          border-radius: var(--radius-sm);
          font-weight: 700;
          font-size: 14px;
          text-align: center;
        }

        .label-badge.secondary {
          background: var(--bg-surface-high);
          border-color: var(--border-color);
          color: var(--text-primary);
        }

        .probability-table {
          margin-top: var(--spacing-xs);
          border-top: 1px solid var(--border-color);
          padding-top: var(--spacing-sm);
        }

        .table-header {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 10px;
          color: var(--text-muted);
          margin-bottom: var(--spacing-xs);
        }

        .table-row {
          display: flex;
          justify-content: space-between;
          padding: 4px 0;
          font-size: 12px;
        }

        .row-class {
          color: var(--text-secondary);
        }

        .row-prob {
          font-family: var(--font-mono);
          font-weight: 600;
          color: var(--text-primary);
        }

        .confidence-metrics {
          display: flex;
          flex-direction: column;
          gap: var(--spacing-md);
          margin-top: var(--spacing-sm);
        }

        .metric-info {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 6px;
        }

        .metric-label {
          font-family: var(--font-mono);
          font-size: 11px;
          color: var(--text-secondary);
        }

        .metric-value {
          font-family: var(--font-mono);
          font-size: 12px;
          font-weight: 700;
        }

        .progress-track {
          height: 4px;
          background: var(--bg-surface-high);
          border-radius: var(--radius-full);
          overflow: hidden;
        }

        .progress-bar {
          height: 100%;
          border-radius: var(--radius-full);
        }

        .progress-bar.cnn {
          background: var(--primary);
          box-shadow: 0 0 8px var(--primary-glow);
        }

        .progress-bar.rf {
          background: var(--text-muted);
        }

        .explanation-text {
          margin-top: var(--spacing-lg);
          padding: var(--spacing-sm);
          font-style: italic;
          color: var(--text-muted);
          font-size: 13px;
          text-align: center;
          line-height: 1.4;
        }

        .bottom-stats {
          margin-top: auto;
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: var(--spacing-sm);
        }

        .stat-card {
          background: var(--bg-surface);
          border: 1px solid var(--border-color);
          border-radius: var(--radius-sm);
          padding: 12px;
        }

        .stat-header {
          display: flex;
          align-items: center;
          gap: 6px;
          margin-bottom: 8px;
        }

        .stat-label {
          font-size: 11px;
          color: var(--text-muted);
          font-weight: 600;
        }

        .stat-value {
          font-family: var(--font-mono);
          font-size: 14px;
          font-weight: 600;
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .stat-value.active {
          color: var(--text-primary);
        }

        .pulse-dot {
          width: 6px;
          height: 6px;
          background: var(--success);
          border-radius: var(--radius-full);
          box-shadow: 0 0 0 rgba(16, 185, 129, 0.4);
          animation: pulse 2s infinite;
        }

        @keyframes pulse {
          0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
          70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
          100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
      `}</style>
    </aside>
  );
};

export default ResultsPanel;
