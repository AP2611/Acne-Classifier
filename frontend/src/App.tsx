import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Sidebar from './components/Sidebar';
import TopNav from './components/TopNav';
import ClassificationCard from './components/ClassificationCard';
import ResultsPanel from './components/ResultsPanel';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [metrics, setMetrics] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<'preview' | 'classification'>('preview');

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/metrics`);
        setMetrics(response.data);
      } catch (error) {
        console.error('Error fetching metrics:', error);
      }
    };
    fetchMetrics();
  }, []);

  const handleClassify = async (modelType: string) => {
    if (!selectedFile) return;

    setIsClassifying(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_BASE_URL}/classify?model_type=${modelType}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResults(response.data);
      setActiveTab('classification');
    } catch (error) {
      console.error('Error classifying image:', error);
    } finally {
      setIsClassifying(false);
    }
  };

  return (
    <div className="app-container">
      <Sidebar />
      
      <main className="main-content">
        <TopNav />
        
        <div className="dashboard-grid">
          <div className="primary-column">
            <ClassificationCard 
              onFileSelect={(file) => {
                setSelectedFile(file);
                const url = URL.createObjectURL(file);
                setPreviewUrl(url);
              }}
              onClassify={handleClassify}
              isClassifying={isClassifying}
              onNewFile={() => setActiveTab('preview')}
            />
          </div>
          
          <ResultsPanel 
            results={results}
            isClassifying={isClassifying}
            metrics={metrics}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            previewUrl={previewUrl}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
