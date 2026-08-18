import React, { useState, useRef } from 'react';
import { Upload, Mic, FileText, AlertCircle, CheckCircle, Loader, X, Volume2, BarChart3, Info, Brain, Target, TrendingUp } from 'lucide-react';

const API_URL = process.env.REACT_APP_API_URL || "https://finalyear226-urtox-api.hf.space";

export default function MutexWithXAI() {
  const [activeTab, setActiveTab] = useState('text');
  const [inputText, setInputText] = useState('');
  const [audioFile, setAudioFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [apiStatus, setApiStatus] = useState(null);
  const [showXAI, setShowXAI] = useState(true);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const fileInputRef = useRef(null);
  const recordingInterval = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const analyzeToxicity = async (input, type) => {
    setIsAnalyzing(true);
    setApiStatus(null);
    try {
      let body;

      if (type === 'text') {
        body = { mode: 'text', text: input };
      } else {
        const audioDataUrl = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.onerror = reject;
          reader.readAsDataURL(input);
        });
        body = { mode: 'audio', audio: audioDataUrl };
      }

      const res = await fetch(`${API_URL}/detect`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        },
        body: JSON.stringify(body)
      });

      if (!res.ok) {
        throw new Error(`API returned ${res.status}`);
      }

      const data = await res.json();
      console.log('API Response:', JSON.stringify(data));
      setResults(data);
      setApiStatus({ type: "live", message: "Live model API connected" });

    } catch (err) {
      console.error(err);
      setResults(null);
      setApiStatus({
        type: "error",
        message: "API unavailable. Check whether the backend is running."
      });
      setIsAnalyzing(false);
      return;
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
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      recordingInterval.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } catch (err) {
      alert('Microphone access denied. Please allow microphone access and try again.');
      console.error(err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setAudioFile(audioBlob);
        analyzeToxicity(audioBlob, 'audio');
        mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      };
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    clearInterval(recordingInterval.current);
  };

  const clearAll = () => {
    setInputText('');
    setAudioFile(null);
    setResults(null);
    setApiStatus(null);
    setRecordingTime(0);
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getShapColor = (score) => {
    if (score > 0.7) return 'bg-merlot';
    if (score > 0.4) return 'bg-orange-500';
    if (score > 0.2) return 'bg-yellow-500';
    return 'bg-forest-mid';
  };

  const getSubLabelColor = (label) => {
    const colors = {
      offensive: 'text-merlot-bright bg-merlot-wash/70 border-orange-500/50',
      hate: 'text-merlot-mid bg-merlot/20 border-merlot-bright/50',
      insult: 'text-merlot-mid bg-merlot-bright/20 border-merlot-bright/50',
      profanity: 'text-merlot-mid bg-merlot-mid/20 border-merlot-bright/50',
      neutral: 'text-forest bg-sand/60 border-forest-soft/50'
    };
    return colors[label] || colors.neutral;
  };

  return (
    <div className="bg-ivory px-5 pb-16 pt-8 sm:px-8">
      <style>{`
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
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .glass-card {
          background: rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .shap-bar { position: relative; overflow: hidden; }
        .shap-bar::before {
          content: '';
          position: absolute;
          top: 0; left: -100%;
          width: 100%; height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
          animation: shimmer 2s infinite;
        }
      `}</style>

      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-20 w-96 h-96 bg-merlot-mid rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-merlot-bright rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      <div className="max-w-7xl mx-auto relative z-10">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-merlot via-merlot-bright to-orange-400 bg-clip-text text-transparent mb-3">
            Multimodal Urdu Toxic Span Detection
          </h1>
          <p className="text-forest-mid text-lg mb-2">
            Detect toxic content in Urdu text, roman urdu, english and audio with AI explainability
          </p>
          <div className="flex items-center justify-center gap-2 text-sm text-forest-soft">
            <Brain className="w-4 h-4" />
            <span>Powered by XAI • SHAP Attributions • Integrated Gradients</span>
          </div>
        </div>

        <div className="flex gap-4 mb-8 justify-center">
          <button
            onClick={() => setActiveTab('text')}
            className={`flex items-center gap-3 px-8 py-4 rounded-2xl font-semibold transition-all duration-300 ${
              activeTab === 'text'
                ? 'bg-gradient-to-r from-merlot to-merlot-bright text-forest-deep shadow-lg scale-105'
                : 'glass-card text-forest-mid hover:bg-white/70'
            }`}
          >
            <FileText className="w-5 h-5" />
            Text Analysis
          </button>
          <button
            onClick={() => setActiveTab('audio')}
            className={`flex items-center gap-3 px-8 py-4 rounded-2xl font-semibold transition-all duration-300 ${
              activeTab === 'audio'
                ? 'bg-gradient-to-r from-forest-mid to-forest-mid text-forest-deep shadow-lg scale-105'
                : 'glass-card text-forest-mid hover:bg-white/70'
            }`}
          >
            <Volume2 className="w-5 h-5" />
            Audio Analysis
          </button>
        </div>

        <div className="glass-card rounded-3xl p-8 shadow-2xl">
          {apiStatus && (
            <div className={`mb-6 rounded-xl border px-4 py-3 text-sm ${
              apiStatus.type === "live"
                ? "border-forest-soft/40 bg-green-500/10 text-green-200"
                : apiStatus.type === "demo"
                  ? "border-yellow-500/40 bg-yellow-500/10 text-yellow-100"
                  : "border-merlot-bright/40 bg-merlot/10 text-red-100"
            }`}>
              {apiStatus.message}
            </div>
          )}

          {activeTab === 'text' ? (
            <form onSubmit={handleTextSubmit} className="space-y-6">
              <div>
                <label className="block text-forest-soft font-semibold mb-3 text-lg flex items-center gap-2">
                  <FileText className="w-5 h-5" />
                  Enter your sentence here
                </label>
                <div className="relative">
                  <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="Enter your sentence here"
                    className="w-full h-32 px-6 py-4 pr-14 bg-cream/50 border-2 border-merlot-bright/30 rounded-2xl 
                             text-forest-deep text-lg placeholder-slate-500 focus:outline-none focus:border-merlot-bright
                             transition-all duration-300 resize-none"
                  />
                  <button
                    type="button"
                    onClick={() => setActiveTab('audio')}
                    className="absolute right-4 top-4 w-10 h-10 bg-merlot hover:bg-merlot rounded-full
                             flex items-center justify-center transition-all duration-300 hover:scale-110"
                    title="Switch to audio input"
                  >
                    <Mic className="w-5 h-5 text-forest-deep" />
                  </button>
                </div>
                <div className="flex justify-between items-center mt-2 text-sm text-forest-soft">
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
                  className="flex-1 bg-gradient-to-r from-merlot to-merlot-bright hover:from-merlot hover:to-merlot-bright
                           disabled:from-sand disabled:to-sand-deep disabled:cursor-not-allowed
                           text-forest-deep font-bold py-4 px-8 rounded-xl text-lg
                           transition-all duration-300 transform hover:scale-[1.02] active:scale-[0.98]
                           shadow-lg hover:shadow-merlot/50 flex items-center justify-center gap-3"
                >
                  {isAnalyzing ? (
                    <><Loader className="w-5 h-5 animate-spin" />Analyzing...</>
                  ) : (
                    <><BarChart3 className="w-5 h-5" />Detect Toxic Spans</>
                  )}
                </button>
                <button
                  type="button"
                  onClick={clearAll}
                  className="bg-sand hover:bg-sand text-forest-deep font-semibold py-4 px-8 rounded-xl
                           transition-all duration-300 transform hover:scale-[1.02] active:scale-[0.98] flex items-center gap-2"
                >
                  <X className="w-5 h-5" />Clear
                </button>
              </div>
            </form>
          ) : (
            <div className="space-y-6">
              <div>
                <label className="block text-forest font-semibold mb-3 text-lg flex items-center gap-2">
                  <Volume2 className="w-5 h-5" />
                  Upload or Record Audio
                </label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    className="glass-card border-2 border-forest-soft/30 rounded-2xl p-8 cursor-pointer
                             hover:border-forest-soft transition-all duration-300 hover:scale-[1.02]"
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="audio/*,.ogg,.mp3,.wav"
                      onChange={handleAudioUpload}
                      className="hidden"
                    />
                    <Upload className="w-12 h-12 text-forest mx-auto mb-4" />
                    <p className="text-center text-forest-mid font-semibold mb-2">Upload Audio File</p>
                    <p className="text-center text-sm text-forest-soft">MP3, WAV, OGG</p>
                    {audioFile && audioFile.name && (
                      <p className="text-center text-forest text-sm mt-4">✓ {audioFile.name}</p>
                    )}
                  </div>

                  <div className="glass-card border-2 border-forest-soft/30 rounded-2xl p-8">
                    <Mic className={`w-12 h-12 mx-auto mb-4 ${isRecording ? 'text-merlot-mid animate-pulse' : 'text-forest'}`} />
                    <p className="text-center text-forest-mid font-semibold mb-2">Record Live</p>
                    {isRecording ? (
                      <>
                        <p className="text-center text-merlot-mid font-mono text-2xl mb-4">{formatTime(recordingTime)}</p>
                        <button
                          onClick={stopRecording}
                          className="w-full bg-merlot hover:bg-red-700 text-forest-deep font-bold py-3 px-6 rounded-xl
                                   transition-all duration-300 flex items-center justify-center gap-2"
                        >
                          <div className="w-4 h-4 bg-white rounded-sm"></div>
                          Stop Recording
                        </button>
                      </>
                    ) : (
                      <button
                        onClick={startRecording}
                        className="w-full bg-forest-mid hover:bg-forest-mid text-forest-deep font-bold py-3 px-6 rounded-xl
                                 transition-all duration-300 flex items-center justify-center gap-2"
                      >
                        <div className="w-4 h-4 bg-merlot rounded-full animate-pulse"></div>
                        Start Recording
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {audioFile && (
                <div className="flex gap-4 mt-4">
                  <button
                    onClick={() => analyzeToxicity(audioFile, 'audio')}
                    disabled={isAnalyzing}
                    className="flex-1 bg-gradient-to-r from-forest-mid to-forest-mid hover:from-forest-mid hover:to-forest-mid
                             disabled:from-sand disabled:to-sand-deep disabled:cursor-not-allowed
                             text-forest-deep font-bold py-4 px-8 rounded-xl text-lg
                             transition-all duration-300 transform hover:scale-[1.02]
                             flex items-center justify-center gap-3"
                  >
                    {isAnalyzing ? (
                      <><Loader className="w-5 h-5 animate-spin" />Analyzing Audio...</>
                    ) : (
                      <><BarChart3 className="w-5 h-5" />Detect Toxic Spans</>
                    )}
                  </button>
                  <button
                    onClick={clearAll}
                    className="bg-sand hover:bg-sand text-forest-deep font-semibold py-4 px-8 rounded-xl
                             transition-all duration-300 flex items-center gap-2"
                  >
                    <X className="w-5 h-5" />Clear
                  </button>
                </div>
              )}
            </div>
          )}

          {results && (
            <div className="mt-8 pt-8 border-t border-merlot-bright/30">
              <div className="space-y-4 mb-6">
                <div className="flex items-center justify-center">
                  {results.isToxic ? (
                    <div className="flex items-center gap-2 bg-merlot/20 border border-merlot-bright/50 rounded-full px-6 py-3">
                      <AlertCircle className="w-6 h-6 text-merlot-mid" />
                      <span className="text-merlot-mid font-bold text-lg">Toxic Content Detected</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 bg-sand/60 border border-forest-soft/50 rounded-full px-6 py-3">
                      <CheckCircle className="w-6 h-6 text-forest" />
                      <span className="text-green-300 font-bold text-lg">Safe Content</span>
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="glass-card rounded-xl p-4 border border-merlot-bright/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Target className="w-5 h-5 text-merlot-mid" />
                      <p className="text-sm text-forest-soft">Overall Confidence</p>
                    </div>
                    <p className="text-3xl font-bold text-forest-soft">
                      {(results.confidence * 100).toFixed(1)}%
                    </p>
                    <div className="w-full bg-sand rounded-full h-2 mt-2">
                      <div 
                        className="bg-gradient-to-r from-merlot to-merlot-mid h-2 rounded-full transition-all duration-500"
                        style={{ width: `${results.confidence * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="glass-card rounded-xl p-4 border border-merlot-bright/20">
                    <div className="flex items-center gap-2 mb-2">
                      <BarChart3 className="w-5 h-5 text-forest" />
                      <p className="text-sm text-forest-soft">Category</p>
                    </div>
                    <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full border ${getSubLabelColor(results.subLabel)}`}>
                      <span className="font-bold text-lg capitalize">{results.subLabel}</span>
                    </div>
                    <p className="text-xs text-forest-soft mt-2">
                      {((results.subLabelConfidence || results.confidence) * 100).toFixed(1)}% confidence
                    </p>
                  </div>

                  {results.toxicSpanCount !== undefined && (
                    <div className="glass-card rounded-xl p-4 border border-merlot-bright/20">
                      <div className="flex items-center gap-2 mb-2">
                        <TrendingUp className="w-5 h-5 text-merlot-bright" />
                        <p className="text-sm text-forest-soft">Toxic Spans Found</p>
                      </div>
                      <p className="text-3xl font-bold text-orange-300">{results.toxicSpanCount}</p>
                      <p className="text-xs text-forest-soft mt-2">Identified using BIO tagging</p>
                    </div>
                  )}
                </div>
              </div>

              {results.transcript && (
                <div className="glass-card rounded-xl p-4 mb-6 border border-forest-soft/20">
                  <p className="text-sm text-forest-soft mb-2 flex items-center gap-2">
                    <FileText className="w-4 h-4" />
                    Transcribed Text
                  </p>
                  <p className="text-xl text-forest">{results.transcript}</p>
                </div>
              )}

              <div className="glass-card rounded-2xl p-6 border border-merlot-bright/20 mb-6">
                <h3 className="text-forest-soft font-semibold mb-4 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5" />
                  Token-Level Analysis
                </h3>
                <div className="text-xl leading-relaxed mb-4">
                  {(results.words || []).map((word, idx) => (
                    <span
                      key={idx}
                      className={`word-fade-in ${word.toxic ? 'toxic-highlight text-red-200 font-bold' : 'text-forest'}`}
                      style={{ animationDelay: `${idx * 0.05}s` }}
                      title={`BIO: ${word.bioTag || 'O'} | Confidence: ${((word.confidence || 0) * 100).toFixed(1)}%`}
                    >
                      {word.text}{' '}
                    </span>
                  ))}
                </div>
                {results.isToxic && (
                  <div className="pt-4 border-t border-merlot-bright/20">
                    <p className="text-sm text-forest-soft flex items-center gap-2">
                      <span className="inline-block w-4 h-4 bg-merlot/30 border border-merlot-bright rounded"></span>
                      Toxic words are highlighted in red with BIO tagging
                    </p>
                  </div>
                )}
              </div>

              {results.xai && (
                <div className="glass-card rounded-2xl p-6 border border-forest-soft/20">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-forest-soft font-semibold flex items-center gap-2">
                      <Brain className="w-5 h-5" />
                      AI Explainability (XAI)
                    </h3>
                    <button
                      onClick={() => setShowXAI(!showXAI)}
                      className="text-sm text-forest-soft hover:text-forest-soft transition-colors"
                    >
                      {showXAI ? 'Hide Details' : 'Show Details'}
                    </button>
                  </div>

                  {showXAI && (
                    <div className="space-y-4">
                      <div className="bg-cream/50 rounded-xl p-4">
                        <p className="text-sm text-forest-soft mb-2 font-semibold">Model Explanation:</p>
                        <p className="text-forest-mid">{results.xai.modelExplanation}</p>
                      </div>

                      {results.xai.topToxicTokens && results.xai.topToxicTokens.length > 0 && (
                        <div>
                          <p className="text-sm text-forest-soft mb-3 font-semibold flex items-center gap-2">
                            <TrendingUp className="w-4 h-4" />
                            Top Toxic Tokens (SHAP Attribution):
                          </p>
                          <div className="space-y-3">
                            {results.xai.topToxicTokens.map((token, idx) => (
                              <div key={idx} className="bg-cream/50 rounded-lg p-3">
                                <div className="flex items-center justify-between mb-2">
                                  <span className="text-forest font-semibold">"{token.token}"</span>
                                  <span className="text-sm text-forest-soft">
                                    SHAP: {(token.attribution * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <div className="w-full bg-sand rounded-full h-3 overflow-hidden shap-bar">
                                  <div 
                                    className={`${getShapColor(token.attribution)} h-3 rounded-full transition-all duration-500`}
                                    style={{ width: `${token.attribution * 100}%` }}
                                  ></div>
                                </div>
                                <p className="text-xs text-forest-soft mt-1">
                                  Confidence: {((token.confidence || token.attribution) * 100).toFixed(1)}%
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {results.xai.integratedGradients !== undefined && (
                        <div className="bg-cream/50 rounded-xl p-4">
                          <p className="text-sm text-forest-soft mb-2 font-semibold">Integrated Gradients Score:</p>
                          <div className="flex items-center gap-4">
                            <div className="flex-1">
                              <div className="w-full bg-sand rounded-full h-3">
                                <div 
                                  className="bg-gradient-to-r from-forest-mid to-merlot-mid h-3 rounded-full transition-all duration-500"
                                  style={{ width: `${results.xai.integratedGradients * 100}%` }}
                                ></div>
                              </div>
                            </div>
                            <span className="text-2xl font-bold text-forest-soft">
                              {(results.xai.integratedGradients * 100).toFixed(1)}%
                            </span>
                          </div>
                          <p className="text-xs text-forest-soft mt-2">
                            Measures feature importance through gradient-based attribution
                          </p>
                        </div>
                      )}

                      <div className="bg-forest/20 border border-forest-soft/30 rounded-xl p-4">
                        <p className="text-sm text-forest-soft mb-2 font-semibold flex items-center gap-2">
                          <Info className="w-4 h-4" />
                          About XAI Methods:
                        </p>
                        <ul className="text-xs text-forest-soft space-y-1 list-disc list-inside">
                          <li><strong>SHAP (SHapley Additive exPlanations):</strong> Shows which words contribute most to toxicity</li>
                          <li><strong>Integrated Gradients:</strong> Measures token importance through gradient analysis</li>
                          <li><strong>BIO Tagging:</strong> B-TOXIC (Begin), I-TOXIC (Inside), O (Outside) span markers</li>
                        </ul>
                      </div>
                    </div>
                  )}
                </div>
              )}

              <div className="mt-6 text-center text-sm text-forest-soft">
                <p>Analyzed using explainable AI-powered toxic span detection</p>
              </div>
            </div>
          )}
        </div>

        <div className="mt-8 text-center">
          <p className="text-forest-soft text-sm">
            Multimodal Urdu Toxic Span Detection System
          </p>
        </div>
      </div>
    </div>
  );
}
