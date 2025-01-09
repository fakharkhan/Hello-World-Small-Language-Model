import React, { useState } from 'react';

const QuestionComponent = () => {
    const [question, setQuestion] = useState('');
    const [conversations, setConversations] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!question.trim()) return;

        const newQuestion = question;
        setQuestion('');
        setLoading(true);
        setError(null);

        // Add user's question immediately
        setConversations(prev => [...prev, {
            type: 'question',
            content: newQuestion,
            timestamp: new Date().toLocaleTimeString()
        }]);

        try {
            const response = await fetch('http://localhost:8000/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: newQuestion }),
            });

            if (!response.ok) {
                throw new Error('Failed to get answer');
            }

            const data = await response.json();
            
            // Add AI's response
            setConversations(prev => [...prev, {
                type: 'answer',
                content: data.answer,
                timestamp: new Date().toLocaleTimeString()
            }]);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto p-4 h-screen flex flex-col">
            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto mb-4 p-4 bg-gray-50 rounded-lg">
                {conversations.length === 0 ? (
                    <div className="text-center text-gray-500 mt-8">
                        Start a conversation by asking a question!
                    </div>
                ) : (
                    <div className="space-y-4">
                        {conversations.map((msg, index) => (
                            <div
                                key={index}
                                className={`flex ${msg.type === 'question' ? 'justify-end' : 'justify-start'}`}
                            >
                                <div
                                    className={`max-w-[70%] rounded-lg p-3 ${
                                        msg.type === 'question'
                                            ? 'bg-indigo-600 text-white'
                                            : 'bg-white border border-gray-200'
                                    }`}
                                >
                                    <div className="text-sm">{msg.content}</div>
                                    <div
                                        className={`text-xs mt-1 ${
                                            msg.type === 'question'
                                                ? 'text-indigo-200'
                                                : 'text-gray-500'
                                        }`}
                                    >
                                        {msg.timestamp}
                                    </div>
                                </div>
                            </div>
                        ))}
                        {loading && (
                            <div className="flex justify-start">
                                <div className="bg-white border border-gray-200 rounded-lg p-3">
                                    <div className="text-gray-500">Typing...</div>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Error Message */}
            {error && (
                <div className="mb-4 text-red-600 text-center">
                    Error: {error}
                </div>
            )}

            {/* Input Form */}
            <form onSubmit={handleSubmit} className="flex gap-2">
                <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    className="flex-1 rounded-lg border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                    placeholder="Type your question here..."
                />
                <button
                    type="submit"
                    disabled={loading || !question.trim()}
                    className="inline-flex justify-center rounded-lg border border-transparent bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {loading ? 'Sending...' : 'Send'}
                </button>
            </form>
        </div>
    );
};

export default QuestionComponent; 