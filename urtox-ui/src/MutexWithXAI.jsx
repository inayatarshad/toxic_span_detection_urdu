import React, { useState, useRef } from 'react';
import { Upload, Mic, FileText, AlertCircle, CheckCircle, Loader, X, Volume2, BarChart3, Info, Brain, Target, TrendingUp } from 'lucide-react';

export default function MutexWithXAI() {
  const [activeTab, setActiveTab] = useState('text');
  const [inputText, setInputText] = useState('');
  const [audioFile, setAudioFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [showXAI, setShowXAI] = useState(true);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const fileInputRef = useRef(null);
  const recordingInterval = useRef(null);
  const API_URL = "https://foraminate-olevia-periodontal.ngrok-free.dev";  // Mock API call - replace with your actual MUTEX-M model endpoint
  const analyzeToxicity = async (input, type) => {
    setIsAnalyzing(true);
    try {
      let body;

      if (type === 'text') {
        body = { mode: 'text', text: input };
      } else {
        const base64 = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result.split(',')[1]);
          reader.onerror = reject;
          reader.readAsDataURL(input);
        });
        body = { mode: 'audio', audio: base64 };
      }

      const res = await fetch(`${API_URL}/detect`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        },
        body: JSON.stringify(body)
      });

      const data = await res.json();
      setResults(data);

    } catch (err) {
      console.error(err);
      alert('API error — is your Colab cell still running?');
    }
    setIsAnalyzing(false);
  };

  const handleTextSubmit = (e) => {
    e.preventDefault();
    if (inputText.trim()) {
      analyzeToxicity(inputText, 'text');
    }
  };

  const handleAudioUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAudioFile(file);
      analyzeToxicity(file, 'audio');
    }
  };

  const startRecording = () => {
    setIsRecording(true);
    setRecordingTime(0);
    recordingInterval.current = setInterval(() => {
      setRecordingTime(prev => prev + 1);
    }, 1000);
  };

  const stopRecording = () => {
    setIsRecording(false);
    clearInterval(recordingInterval.current);
    analyzeToxicity(new Blob(), 'audio');
  };

  const clearAll = () => {
    setInputText('');
    setAudioFile(null);
    setResults(null);
    setRecordingTime(0);
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getShapColor = (score) => {
    if (score > 0.7) return 'bg-red-500';
    if (score > 0.4) return 'bg-orange-500';
    if (score > 0.2) return 'bg-yellow-500';
    return 'bg-blue-500';
  };

  const getSubLabelColor = (label) => {
    const colors = {
      offensive: 'text-orange-400 bg-orange-500/20 border-orange-500/50',
      hate: 'text-red-400 bg-red-500/20 border-red-500/50',
      insult: 'text-pink-400 bg-pink-500/20 border-pink-500/50',
      profanity: 'text-purple-400 bg-purple-500/20 border-purple-500/50',
      neutral: 'text-green-400 bg-green-500/20 border-green-500/50'
    };
    return colors[label] || colors.neutral;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-indigo-900 to-slate-900 p-6">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');
        
        @keyframes pulse-ring {
          0% { transform: scale(0.95); opacity: 1; }
          100% { transform: scale(1.2); opacity: 0; }
        }
        
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        
        .toxic-highlight {
          background: linear-gradient(120deg, rgba(239, 68, 68, 0.3), rgba(220, 38, 38, 0.3));
          padding: 2px 6px;
          border-radius: 4px;
          border-bottom: 2px solid #ef4444;
          position: relative;
        }
        
        .toxic-highlight::before {
          content: '';
          position: absolute;
          inset: -2px;
          border-radius: 6px;
          background: linear-gradient(45deg, #ef4444, #dc2626);
          opacity: 0.2;
          z-index: -1;
          animation: pulse-ring 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        
        .word-fade-in {
          animation: fadeInUp 0.4s ease-out forwards;
          opacity: 0;
        }
        
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .glass-card {
          background: rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .shap-bar {
          position: relative;
          overflow: hidden;
        }
        
        .shap-bar::before {
          content: '';
          position: absolute;
          top: 0;
          left: -100%;
          width: 100%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
          animation: shimmer 2s infinite;
        }
      `}</style>

      {/* Floating Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-20 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      <div className="max-w-7xl mx-auto relative z-10">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-orange-400 bg-clip-text text-transparent mb-3">
            Multimodal Urdu Toxic Span Detection
          </h1>
          <p className="text-slate-300 text-lg mb-2">
            Detect toxic content in Urdu text and audio with AI explainability
          </p>
          <div className="flex items-center justify-center gap-2 text-sm text-slate-400">
            <Brain className="w-4 h-4" />
            <span>Powered by XAI • SHAP Attributions • Integrated Gradients</span>
          </div>
        </div>

        {/* Mode Selector */}
        <div className="flex gap-4 mb-8 justify-center">
          <button
            onClick={() => setActiveTab('text')}
            className={`flex items-center gap-3 px-8 py-4 rounded-2xl font-semibold transition-all duration-300 ${
              activeTab === 'text'
                ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-lg scale-105'
                : 'glass-card text-slate-300 hover:bg-white/10'
            }`}
          >
            <FileText className="w-5 h-5" />
            Text Analysis
          </button>
          <button
            onClick={() => setActiveTab('audio')}
            className={`flex items-center gap-3 px-8 py-4 rounded-2xl font-semibold transition-all duration-300 ${
              activeTab === 'audio'
                ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg scale-105'
                : 'glass-card text-slate-300 hover:bg-white/10'
            }`}
          >
            <Volume2 className="w-5 h-5" />
            Audio Analysis
          </button>
        </div>

        {/* Main Content */}
        <div className="glass-card rounded-3xl p-8 shadow-2xl">
          {activeTab === 'text' ? (
            <form onSubmit={handleTextSubmit} className="space-y-6">
              <div>
                <label className="block text-purple-300 font-semibold mb-3 text-lg flex items-center gap-2">
                  <FileText className="w-5 h-5" />
                  Enter your sentence here
                </label>
                <div className="relative">
                  <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="Enter your sentence here"
                    className="w-full h-32 px-6 py-4 pr-14 bg-slate-900/50 border-2 border-purple-500/30 rounded-2xl 
                             text-white text-lg placeholder-slate-500 focus:outline-none focus:border-purple-400
                             transition-all duration-300 resize-none"
                  />
                  <button
                    type="button"
                    onClick={() => setActiveTab('audio')}
                    className="absolute right-4 top-4 w-10 h-10 bg-red-500 hover:bg-red-600 rounded-full
                             flex items-center justify-center transition-all duration-300 hover:scale-110"
                    title="Switch to audio input"
                  >
                    <Mic className="w-5 h-5 text-white" />
                  </button>
                </div>
                <div className="flex justify-between items-center mt-2 text-sm text-slate-400">
                  <span>{inputText.length} characters</span>
                  <span className="flex items-center gap-2">
                    <Info className="w-4 h-4" />
                    Supports both Nastaliq and Roman Urdu
                  </span>
                </div>
              </div>

              <div className="flex gap-4">
                <button
                  type="submit"
                  disabled={!inputText.trim() || isAnalyzing}
                  className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700
                           disabled:from-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed
                           text-white font-bold py-4 px-8 rounded-xl text-lg
                           transition-all duration-300 transform hover:scale-[1.02] active:scale-[0.98]
                           shadow-lg hover:shadow-purple-500/50 flex items-center justify-center gap-3"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader className="w-5 h-5 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <BarChart3 className="w-5 h-5" />
                      Detect Toxic Spans
                    </>
                  )}
                </button>
                
                <button
                  type="button"
                  onClick={clearAll}
                  className="bg-slate-700 hover:bg-slate-600 text-white font-semibold py-4 px-8 rounded-xl
                           transition-all duration-300 transform hover:scale-[1.02] active:scale-[0.98] flex items-center gap-2"
                >
                  <X className="w-5 h-5" />
                  Clear
                </button>
              </div>
            </form>
          ) : (
            <div className="space-y-6">
              <div>
                <label className="block text-blue-300 font-semibold mb-3 text-lg flex items-center gap-2">
                  <Volume2 className="w-5 h-5" />
                  Upload or Record Audio
                </label>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    className="glass-card border-2 border-blue-500/30 rounded-2xl p-8 cursor-pointer
                             hover:border-blue-400 transition-all duration-300 hover:scale-[1.02]"
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="audio/*,.ogg,.mp3,.wav"
                      onChange={handleAudioUpload}
                      className="hidden"
                    />
                    <Upload className="w-12 h-12 text-blue-400 mx-auto mb-4" />
                    <p className="text-center text-slate-300 font-semibold mb-2">Upload Audio File</p>
                    <p className="text-center text-sm text-slate-500">MP3, WAV, OGG</p>
                    {audioFile && (
                      <p className="text-center text-green-400 text-sm mt-4">✓ {audioFile.name}</p>
                    )}
                  </div>

                  <div className="glass-card border-2 border-blue-500/30 rounded-2xl p-8">
                    <Mic className={`w-12 h-12 mx-auto mb-4 ${isRecording ? 'text-red-400 animate-pulse' : 'text-blue-400'}`} />
                    <p className="text-center text-slate-300 font-semibold mb-2">Record Live</p>
                    {isRecording ? (
                      <>
                        <p className="text-center text-red-400 font-mono text-2xl mb-4">{formatTime(recordingTime)}</p>
                        <button
                          onClick={stopRecording}
                          className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-6 rounded-xl
                                   transition-all duration-300 flex items-center justify-center gap-2"
                        >
                          <div className="w-4 h-4 bg-white rounded-sm"></div>
                          Stop Recording
                        </button>
                      </>
                    ) : (
                      <button
                        onClick={startRecording}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-xl
                                 transition-all duration-300 flex items-center justify-center gap-2"
                      >
                        <div className="w-4 h-4 bg-red-500 rounded-full animate-pulse"></div>
                        Start Recording
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Results Section */}
          {results && (
            <div className="mt-8 pt-8 border-t border-purple-500/30">
              {/* Status Header with Confidence and Sub-label */}
              <div className="space-y-4 mb-6">
                <div className="flex items-center justify-center">
                  {results.isToxic ? (
                    <div className="flex items-center gap-2 bg-red-500/20 border border-red-500/50 rounded-full px-6 py-3">
                      <AlertCircle className="w-6 h-6 text-red-400" />
                      <span className="text-red-300 font-bold text-lg">Toxic Content Detected</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 bg-green-500/20 border border-green-500/50 rounded-full px-6 py-3">
                      <CheckCircle className="w-6 h-6 text-green-400" />
                      <span className="text-green-300 font-bold text-lg">Safe Content</span>
                    </div>
                  )}
                </div>

                {/* Confidence and Sub-label Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="glass-card rounded-xl p-4 border border-purple-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Target className="w-5 h-5 text-purple-400" />
                      <p className="text-sm text-slate-400">Overall Confidence</p>
                    </div>
                    <p className="text-3xl font-bold text-purple-300">
                      {(results.confidence * 100).toFixed(1)}%
                    </p>
                    <div className="w-full bg-slate-700 rounded-full h-2 mt-2">
                      <div 
                        className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${results.confidence * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="glass-card rounded-xl p-4 border border-purple-500/20">
                    <div className="flex items-center gap-2 mb-2">
                      <BarChart3 className="w-5 h-5 text-blue-400" />
                      <p className="text-sm text-slate-400">Category</p>
                    </div>
                    <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full border ${getSubLabelColor(results.subLabel)}`}>
                      <span className="font-bold text-lg capitalize">{results.subLabel}</span>
                    </div>
                    <p className="text-xs text-slate-500 mt-2">
                      {(results.subLabelConfidence * 100).toFixed(1)}% confidence
                    </p>
                  </div>

                  {results.toxicSpanCount !== undefined && (
                    <div className="glass-card rounded-xl p-4 border border-purple-500/20">
                      <div className="flex items-center gap-2 mb-2">
                        <TrendingUp className="w-5 h-5 text-orange-400" />
                        <p className="text-sm text-slate-400">Toxic Spans Found</p>
                      </div>
                      <p className="text-3xl font-bold text-orange-300">
                        {results.toxicSpanCount}
                      </p>
                      <p className="text-xs text-slate-500 mt-2">
                        Identified using BIO tagging
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* Audio Transcript */}
              {results.transcript && (
                <div className="glass-card rounded-xl p-4 mb-6 border border-blue-500/20">
                  <p className="text-sm text-slate-400 mb-2 flex items-center gap-2">
                    <FileText className="w-4 h-4" />
                    Transcribed Text
                  </p>
                  <p className="text-xl text-slate-200">{results.transcript}</p>
                </div>
              )}

              {/* Analyzed Text with Toxic Spans */}
              <div className="glass-card rounded-2xl p-6 border border-purple-500/20 mb-6">
                <h3 className="text-purple-300 font-semibold mb-4 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5" />
                  Token-Level Analysis
                </h3>
                <div className="text-xl leading-relaxed mb-4">
                  {results.words.map((word, idx) => (
                    <span
                      key={word.id}
                      className={`word-fade-in ${word.toxic ? 'toxic-highlight text-red-200 font-bold' : 'text-slate-200'}`}
                      style={{ animationDelay: `${idx * 0.05}s` }}
                      title={`BIO: ${word.bioTag || 'O'} | Confidence: ${((word.confidence || 0) * 100).toFixed(1)}%`}
                    >
                      {word.text}
                    </span>
                  ))}
                </div>
                
                {results.isToxic && (
                  <div className="pt-4 border-t border-purple-500/20">
                    <p className="text-sm text-slate-400 flex items-center gap-2">
                      <span className="inline-block w-4 h-4 bg-red-500/30 border border-red-500 rounded"></span>
                      Toxic words are highlighted in red with BIO tagging
                    </p>
                  </div>
                )}
              </div>

              {/* XAI Explainability Section */}
              {results.xai && (
                <div className="glass-card rounded-2xl p-6 border border-indigo-500/20">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-indigo-300 font-semibold flex items-center gap-2">
                      <Brain className="w-5 h-5" />
                      AI Explainability (XAI)
                    </h3>
                    <button
                      onClick={() => setShowXAI(!showXAI)}
                      className="text-sm text-indigo-400 hover:text-indigo-300 transition-colors"
                    >
                      {showXAI ? 'Hide Details' : 'Show Details'}
                    </button>
                  </div>

                  {showXAI && (
                    <div className="space-y-4">
                      {/* Model Explanation */}
                      <div className="bg-slate-900/50 rounded-xl p-4">
                        <p className="text-sm text-slate-400 mb-2 font-semibold">Model Explanation:</p>
                        <p className="text-slate-300">{results.xai.modelExplanation}</p>
                      </div>

                      {/* SHAP Attribution Scores */}
                      {results.xai.topToxicTokens && results.xai.topToxicTokens.length > 0 && (
                        <div>
                          <p className="text-sm text-slate-400 mb-3 font-semibold flex items-center gap-2">
                            <TrendingUp className="w-4 h-4" />
                            Top Toxic Tokens (SHAP Attribution):
                          </p>
                          <div className="space-y-3">
                            {results.xai.topToxicTokens.map((token, idx) => (
                              <div key={idx} className="bg-slate-900/50 rounded-lg p-3">
                                <div className="flex items-center justify-between mb-2">
                                  <span className="text-slate-200 font-semibold">"{token.token}"</span>
                                  <span className="text-sm text-slate-400">
                                    SHAP: {(token.attribution * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <div className="w-full bg-slate-700 rounded-full h-3 overflow-hidden shap-bar">
                                  <div 
                                    className={`${getShapColor(token.attribution)} h-3 rounded-full transition-all duration-500`}
                                    style={{ width: `${token.attribution * 100}%` }}
                                  ></div>
                                </div>
                                <p className="text-xs text-slate-500 mt-1">
                                  Confidence: {(token.confidence * 100).toFixed(1)}%
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Integrated Gradients */}
                      <div className="bg-slate-900/50 rounded-xl p-4">
                        <p className="text-sm text-slate-400 mb-2 font-semibold">Integrated Gradients Score:</p>
                        <div className="flex items-center gap-4">
                          <div className="flex-1">
                            <div className="w-full bg-slate-700 rounded-full h-3">
                              <div 
                                className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-500"
                                style={{ width: `${results.xai.integratedGradients * 100}%` }}
                              ></div>
                            </div>
                          </div>
                          <span className="text-2xl font-bold text-purple-300">
                            {(results.xai.integratedGradients * 100).toFixed(1)}%
                          </span>
                        </div>
                        <p className="text-xs text-slate-500 mt-2">
                          Measures feature importance through gradient-based attribution
                        </p>
                      </div>

                      {/* XAI Method Info */}
                      <div className="bg-indigo-900/20 border border-indigo-500/30 rounded-xl p-4">
                        <p className="text-sm text-indigo-300 mb-2 font-semibold flex items-center gap-2">
                          <Info className="w-4 h-4" />
                          About XAI Methods:
                        </p>
                        <ul className="text-xs text-slate-400 space-y-1 list-disc list-inside">
                          <li><strong>SHAP (SHapley Additive exPlanations):</strong> Shows which words contribute most to toxicity</li>
                          <li><strong>Integrated Gradients:</strong> Measures token importance through gradient analysis</li>
                          <li><strong>BIO Tagging:</strong> B-TOXIC (Begin), I-TOXIC (Inside), O (Outside) span markers</li>
                        </ul>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Footer Info */}
              <div className="mt-6 text-center text-sm text-slate-400">
                <p>Analyzed using explainable AI-powered toxic span detection</p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-8 text-center">
          <p className="text-slate-400 text-sm">
            Multimodal Urdu Toxic Span Detection System
          </p>
        </div>
      </div>
    </div>
  );
}