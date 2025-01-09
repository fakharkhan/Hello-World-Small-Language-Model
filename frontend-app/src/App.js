import React from 'react';
import QuestionComponent from './components/QuestionComponent';
import './App.css';

function App() {
  return (
    <div className="App min-h-screen bg-gray-100">
      <header className="bg-white shadow-sm">
        <div className="max-w-4xl mx-auto py-4 px-4">
          <h1 className="text-2xl font-semibold text-gray-900">AI Chat Assistant</h1>
        </div>
      </header>
      <main className="mt-4">
        <QuestionComponent />
      </main>
    </div>
  );
}

export default App;
