import React, { useState, useRef } from 'react';
import { Upload, FileCode, Zap, ChevronDown } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface ClassificationCardProps {
  onFileSelect: (file: File) => void;
  onClassify: (model: string) => void;
  isClassifying: boolean;
  onNewFile: () => void;
}

const ClassificationCard: React.FC<ClassificationCardProps> = ({ onFileSelect, onClassify, isClassifying, onNewFile }) => {
  const [selectedModel, setSelectedModel] = useState('cnn');
  const [preview, setPreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      onFileSelect(file);
      onNewFile();
      const reader = new FileReader();
      reader.onload = (event) => setPreview(event.target?.result as string);
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="card-container">
      <div className="card-header">
        <h1>Image Classification</h1>
        <p>Deploy high-performance computer vision models. Upload your assets, select your architecture, and generate real-time inference reports.</p>
      </div>

      <motion.div 
        className="main-card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="card-tag">
          <FileCode size={14} />
          <span className="technical-label">New Classification</span>
        </div>

        <div 
          className="drop-zone"
          onClick={() => fileInputRef.current?.click()}
        >
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            hidden 
            accept="image/*"
          />
          
          <AnimatePresence mode="wait">
            {preview ? (
              <motion.div 
                key="preview"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="image-preview"
              >
                <img src={preview} alt="Upload Preview" />
              </motion.div>
            ) : (
              <motion.div 
                key="placeholder"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="placeholder"
              >
                <div className="upload-icon">
                  <Upload size={24} />
                </div>
                <h3>Drag & drop your images here</h3>
                <p>Supports JPG, PNG or DICOM (Max 20MB)</p>
                <button className="browse-btn">Or browse files</button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="config-section">
          <label className="technical-label">Model Configuration</label>
          <div className="model-selector">
            <div className="selected-model">
              {selectedModel === 'cnn' ? 'CNN (Convolutional Neural Network)' : 'RF (Random Forest Baseline)'}
            </div>
            <ChevronDown size={18} />
            <select 
              value={selectedModel} 
              onChange={(e) => setSelectedModel(e.target.value)}
              className="hidden-select"
            >
              <option value="cnn">CNN (Convolutional Neural Network)</option>
              <option value="rf">RF (Random Forest Baseline)</option>
              <option value="both">Compare Both Models</option>
            </select>
          </div>
        </div>

        <button 
          className={`classify-btn ${isClassifying ? 'loading' : ''}`}
          onClick={() => onClassify(selectedModel)}
          disabled={!preview || isClassifying}
        >
          <Zap size={18} fill="currentColor" />
          <span>{isClassifying ? 'Classifying...' : 'Classify Image'}</span>
        </button>
      </motion.div>

      <style jsx>{`
        .card-container {
          max-width: 800px;
        }

        .card-header h1 {
          font-size: 24px;
          margin-bottom: var(--spacing-xs);
        }

        .card-header p {
          color: var(--text-secondary);
          font-size: 14px;
          line-height: 1.5;
          margin-bottom: var(--spacing-lg);
          max-width: 600px;
        }

        .main-card {
          background: var(--bg-surface);
          border: 1px solid var(--border-color);
          border-radius: var(--radius-md);
          padding: var(--spacing-md);
          position: relative;
        }

        .card-tag {
          display: flex;
          align-items: center;
          gap: var(--spacing-xs);
          color: var(--text-primary);
          margin-bottom: var(--spacing-md);
        }

        .drop-zone {
          border: 2px dashed var(--bg-surface-highest);
          border-radius: var(--radius-md);
          height: 280px;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-direction: column;
          cursor: pointer;
          transition: border-color 0.2s ease;
          overflow: hidden;
          background: rgba(255, 255, 255, 0.02);
        }

        .drop-zone:hover {
          border-color: var(--primary);
          background: rgba(99, 102, 241, 0.05);
        }

        .placeholder {
          text-align: center;
        }

        .upload-icon {
          width: 48px;
          height: 48px;
          background: var(--bg-surface-high);
          border-radius: var(--radius-sm);
          display: flex;
          align-items: center;
          justify-content: center;
          margin: 0 auto var(--spacing-sm);
          color: var(--text-secondary);
        }

        .drop-zone h3 {
          font-size: 16px;
          margin-bottom: 4px;
        }

        .drop-zone p {
          font-size: 13px;
          color: var(--text-muted);
          margin-bottom: var(--spacing-md);
        }

        .browse-btn {
          background: transparent;
          color: var(--primary);
          font-weight: 600;
          font-size: 14px;
          text-decoration: underline;
        }

        .image-preview {
          width: 100%;
          height: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: var(--spacing-md);
        }

        .image-preview img {
          max-width: 100%;
          max-height: 100%;
          object-fit: contain;
          border-radius: var(--radius-sm);
        }

        .config-section {
          margin-top: var(--spacing-lg);
          margin-bottom: var(--spacing-md);
        }

        .config-section label {
          display: block;
          margin-bottom: var(--spacing-xs);
          color: var(--text-muted);
        }

        .model-selector {
          background: var(--bg-surface-high);
          border: 1px solid var(--border-color);
          border-radius: var(--radius-sm);
          padding: 12px 16px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          position: relative;
          cursor: pointer;
        }

        .hidden-select {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          opacity: 0;
          cursor: pointer;
        }

        .selected-model {
          font-size: 14px;
          font-weight: 600;
        }

        .classify-btn {
          width: 100%;
          background: var(--primary);
          color: white;
          padding: 14px;
          border-radius: var(--radius-sm);
          font-weight: 700;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: var(--spacing-xs);
          font-size: 15px;
          box-shadow: 0 4px 20px var(--primary-glow);
        }

        .classify-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
          box-shadow: none;
        }

        .classify-btn.loading {
          opacity: 0.8;
        }
      `}</style>
    </div>
  );
};

export default ClassificationCard;
