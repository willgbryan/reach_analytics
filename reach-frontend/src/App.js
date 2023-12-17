import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import UserDiv from './main_page';
import MarketingPage from './marketing_page';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<UserDiv />} />
        <Route path="/marketing-site" element={<MarketingPage />} />
      </Routes>
    </Router>
  );
}

export default App;
