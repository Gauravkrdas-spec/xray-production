import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <BrowserRouter>
    <Routes>
      <Route path="/analyzer" element={<App />} />
      <Route path="*" element={
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
          flexDirection: 'column',
          gap: 16,
          background: '#0d0f14',
          color: '#fff'
        }}>
          <h2>Page not found</h2>
          <a href="/analyzer"
            style={{ color: '#4A90D9' }}>
            Go to Analyzer
          </a>
        </div>
      } />
    </Routes>
  </BrowserRouter>
);