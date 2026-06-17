"""
Complete AI-Powered Spam Detection System
Single file - No uvicorn.run required
Dark Theme with Vibrant Colors
"""

import os
import sqlite3
import pickle
import socket
import webbrowser
from datetime import datetime
from contextlib import contextmanager
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATABASE_PATH = "spam_detection.db"
MODELS_DIR = "models"
PORT = 8000

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================
# Database
# ============================================================
class Database:
    def __init__(self):
        self.db_path = DATABASE_PATH
        self._init_db()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def save_scan(self, message: str, prediction: str, confidence: float):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO history (message, prediction, confidence) VALUES (?, ?, ?)",
                (message, prediction, confidence)
            )
            conn.commit()
            return cursor.lastrowid

    def get_history(self, limit: int = 50):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM history ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_history(self, id: int):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM history WHERE id = ?", (id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_stats(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            total = cursor.execute("SELECT COUNT(*) FROM history").fetchone()[0]
            spam = cursor.execute("SELECT COUNT(*) FROM history WHERE prediction = 'spam'").fetchone()[0]
            legitimate = cursor.execute("SELECT COUNT(*) FROM history WHERE prediction = 'legitimate'").fetchone()[0]
            return {"total": total, "spam": spam, "legitimate": legitimate}

db = Database()

# ============================================================
# AI Model
# ============================================================
class SpamClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.model_file = os.path.join(MODELS_DIR, "spam_model.pkl")
        self.vectorizer_file = os.path.join(MODELS_DIR, "vectorizer.pkl")
        self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(self.model_file) and os.path.exists(self.vectorizer_file):
            try:
                with open(self.model_file, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.vectorizer_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                print("✅ Model loaded successfully")
                return
            except:
                print("⚠️ Error loading model, training new...")

        self._train_model()

    def _train_model(self):
        print("🤖 Training spam detection model...")
        
        # Training data
        data = [
            ("Congratulations! You won a free iPhone. Click here", "spam"),
            ("URGENT: Your bank account has been compromised", "spam"),
            ("You have won $1000 gift card. Claim now", "spam"),
            ("Limited time offer! 90% discount today only", "spam"),
            ("WINNER! You've been selected for grand prize", "spam"),
            ("Free money! Click here to get $500", "spam"),
            ("Your account needs verification immediately", "spam"),
            ("You are the lucky winner of our contest", "spam"),
            ("Special offer just for you, don't miss out", "spam"),
            ("Congratulations! You've won a free vacation", "spam"),
            ("Your password needs to be reset", "spam"),
            ("Security alert: Unusual activity detected", "spam"),
            ("Verify your account now to avoid suspension", "spam"),
            ("You have received a payment of $1000", "spam"),
            ("Claim your free gift now!", "spam"),
            ("Meeting at 3pm tomorrow in conference room B", "legitimate"),
            ("Hey, are you coming to the party tonight?", "legitimate"),
            ("Please review the attached document", "legitimate"),
            ("Can you send me the report by EOD?", "legitimate"),
            ("Let's schedule a call for next week", "legitimate"),
            ("I've sent you the files you requested", "legitimate"),
            ("The meeting has been rescheduled to Friday", "legitimate"),
            ("Please find the updated document attached", "legitimate"),
            ("Can we discuss the project timeline?", "legitimate"),
            ("Thank you for your email, I'll get back soon", "legitimate"),
            ("The team meeting is at 2 PM tomorrow", "legitimate"),
            ("Please review the quarterly report", "legitimate"),
            ("Your order has been shipped successfully", "legitimate"),
            ("Thank you for your purchase", "legitimate"),
            ("Your subscription has been renewed", "legitimate"),
        ]
        
        df = pd.DataFrame(data, columns=["text", "label"])
        df["label"] = df["label"].map({"spam": 1, "legitimate": 0})
        
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        X = self.vectorizer.fit_transform(df["text"])
        y = df["label"]
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X, y)
        
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_file, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print("✅ Model trained and saved successfully!")

    def predict(self, text: str):
        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        label = "spam" if prediction == 1 else "legitimate"
        
        return {
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "spam": round(probabilities[1] * 100, 2),
                "legitimate": round(probabilities[0] * 100, 2)
            }
        }

# Initialize model
spam_detector = SpamClassifier()

# ============================================================
# HTML Content with Dark Theme
# ============================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛡️ Spam Detection System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0a0a;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            background-image: 
                radial-gradient(ellipse at 20% 50%, rgba(100, 181, 246, 0.05) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 50%, rgba(0, 255, 0, 0.05) 0%, transparent 50%),
                linear-gradient(180deg, #0a0a0a 0%, #0a0a1a 100%);
        }
        
        .container {
            background: rgba(20, 20, 30, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            padding: 40px;
            max-width: 850px;
            width: 100%;
            box-shadow: 
                0 20px 60px rgba(0, 0, 0, 0.8),
                0 0 40px rgba(100, 181, 246, 0.05),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        h1 {
            color: #fff;
            font-size: 32px;
            font-weight: 800;
            background: linear-gradient(135deg, #4fc3f7, #29b6f6, #0288d1, #4fc3f7);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 40px rgba(79, 195, 247, 0.2);
            animation: gradientShift 3s ease-in-out infinite;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .status-badge {
            background: rgba(255, 255, 255, 0.05);
            padding: 6px 16px;
            border-radius: 20px;
            color: #4fc3f7;
            font-size: 13px;
            font-weight: 600;
            border: 1px solid rgba(79, 195, 247, 0.2);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-badge .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4fc3f7;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.3; }
            100% { opacity: 1; }
        }
        
        .subtitle {
            color: #888;
            margin-bottom: 30px;
            font-size: 15px;
            border-left: 3px solid #4fc3f7;
            padding-left: 15px;
        }
        
        textarea {
            width: 100%;
            padding: 16px;
            background: rgba(255, 255, 255, 0.05);
            border: 2px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            font-size: 16px;
            resize: vertical;
            min-height: 120px;
            font-family: inherit;
            transition: all 0.3s;
            color: #e0e0e0;
        }
        
        textarea:focus {
            outline: none;
            border-color: #4fc3f7;
            background: rgba(79, 195, 247, 0.05);
            box-shadow: 0 0 30px rgba(79, 195, 247, 0.05);
        }
        
        textarea::placeholder {
            color: #555;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 12px 35px;
            border: none;
            border-radius: 12px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #4fc3f7, #0288d1);
            color: white;
            box-shadow: 0 4px 20px rgba(79, 195, 247, 0.3);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 30px rgba(79, 195, 247, 0.4);
        }
        
        .btn-primary:active {
            transform: scale(0.97);
        }
        
        .btn-secondary {
            background: rgba(255, 255, 255, 0.08);
            color: #aaa;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.12);
            color: #fff;
        }
        
        .btn-success {
            background: linear-gradient(135deg, #4caf50, #2e7d32);
            color: white;
            box-shadow: 0 4px 20px rgba(76, 175, 80, 0.3);
        }
        
        .btn-success:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 30px rgba(76, 175, 80, 0.4);
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #ff4444, #cc0000);
            color: white;
            padding: 4px 14px;
            font-size: 12px;
            border-radius: 8px;
        }
        
        .btn-danger:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 20px rgba(255, 68, 68, 0.3);
        }
        
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 14px;
            display: none;
            animation: slideIn 0.4s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result.spam {
            background: rgba(255, 0, 0, 0.15);
            border: 2px solid rgba(255, 0, 0, 0.3);
            display: block;
            box-shadow: 0 0 40px rgba(255, 0, 0, 0.05);
        }
        
        .result.legitimate {
            background: rgba(76, 175, 80, 0.15);
            border: 2px solid rgba(76, 175, 80, 0.3);
            display: block;
            box-shadow: 0 0 40px rgba(76, 175, 80, 0.05);
        }
        
        .result h3 {
            color: #fff;
            margin-bottom: 10px;
            font-size: 16px;
        }
        
        .result .prediction {
            font-size: 28px;
            font-weight: 800;
            text-transform: uppercase;
        }
        
        .result.spam .prediction {
            color: #ff4444;
            text-shadow: 0 0 30px rgba(255, 68, 68, 0.3);
        }
        
        .result.legitimate .prediction {
            color: #4caf50;
            text-shadow: 0 0 30px rgba(76, 175, 80, 0.3);
        }
        
        .result .confidence {
            margin-top: 5px;
            color: #aaa;
        }
        
        .result .probabilities {
            margin-top: 8px;
            font-size: 13px;
            color: #888;
        }
        
        .result .timestamp {
            margin-top: 8px;
            font-size: 12px;
            color: #666;
        }
        
        .loading {
            display: none;
            margin-top: 15px;
            color: #888;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(79, 195, 247, 0.1);
            border-top: 3px solid #4fc3f7;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            vertical-align: middle;
            margin-right: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 30px;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.03);
            padding: 18px;
            border-radius: 14px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.3s;
        }
        
        .stat-card:hover {
            background: rgba(255, 255, 255, 0.05);
            transform: translateY(-2px);
        }
        
        .stat-card .number {
            font-size: 32px;
            font-weight: 800;
            color: #fff;
        }
        
        .stat-card .label {
            color: #666;
            font-size: 13px;
            margin-top: 5px;
        }
        
        .stat-card .number.spam-color {
            color: #ff4444;
        }
        
        .stat-card .number.legitimate-color {
            color: #4caf50;
        }
        
        .stat-card .number.total-color {
            color: #4fc3f7;
        }
        
        .history-section {
            margin-top: 30px;
        }
        
        .history-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .history-section h3 {
            color: #e0e0e0;
            font-size: 18px;
        }
        
        .history-container {
            max-height: 300px;
            overflow-y: auto;
            border-radius: 12px;
            background: rgba(0, 0, 0, 0.2);
            padding: 5px;
        }
        
        .history-container::-webkit-scrollbar {
            width: 6px;
        }
        
        .history-container::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 10px;
        }
        
        .history-container::-webkit-scrollbar-thumb {
            background: rgba(79, 195, 247, 0.3);
            border-radius: 10px;
        }
        
        .history-item {
            padding: 12px 15px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 14px;
            flex-wrap: wrap;
            gap: 8px;
            transition: all 0.3s;
            border-radius: 8px;
        }
        
        .history-item:hover {
            background: rgba(255, 255, 255, 0.03);
        }
        
        .history-item .message-text {
            flex: 1;
            min-width: 150px;
            color: #ccc;
        }
        
        .history-item .badge {
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }
        
        .badge.spam {
            background: rgba(255, 0, 0, 0.2);
            color: #ff4444;
            border: 1px solid rgba(255, 0, 0, 0.2);
        }
        
        .badge.legitimate {
            background: rgba(76, 175, 80, 0.2);
            color: #4caf50;
            border: 1px solid rgba(76, 175, 80, 0.2);
        }
        
        .history-item .date {
            font-size: 11px;
            color: #555;
        }
        
        .no-history {
            color: #555;
            font-size: 14px;
            text-align: center;
            padding: 30px;
        }
        
        .toast {
            position: fixed;
            bottom: 30px;
            right: 30px;
            padding: 15px 25px;
            border-radius: 12px;
            z-index: 1000;
            animation: slideIn 0.3s ease;
            max-width: 400px;
            font-weight: 600;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        }
        
        .toast.success {
            background: linear-gradient(135deg, #2e7d32, #4caf50);
            color: white;
            border: 1px solid rgba(76, 175, 80, 0.3);
        }
        
        .toast.error {
            background: linear-gradient(135deg, #cc0000, #ff4444);
            color: white;
            border: 1px solid rgba(255, 68, 68, 0.3);
        }
        
        .toast.info {
            background: linear-gradient(135deg, #01579b, #4fc3f7);
            color: white;
            border: 1px solid rgba(79, 195, 247, 0.3);
        }
        
        .action-buttons {
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .refresh-btn {
            background: rgba(79, 195, 247, 0.15);
            color: #4fc3f7;
            border: 1px solid rgba(79, 195, 247, 0.2);
            padding: 8px 20px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .refresh-btn:hover {
            background: rgba(79, 195, 247, 0.25);
            transform: translateY(-2px);
        }
        
        .refresh-btn:active {
            transform: scale(0.95);
        }
        
        .confidence-bar {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            margin-top: 10px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.8s ease;
        }
        
        .result.spam .confidence-fill {
            background: linear-gradient(90deg, #ff4444, #ff6b6b);
        }
        
        .result.legitimate .confidence-fill {
            background: linear-gradient(90deg, #4caf50, #81c784);
        }
        
        @media (max-width: 600px) {
            .container {
                padding: 20px;
            }
            
            .stats {
                grid-template-columns: 1fr;
            }
            
            .history-item {
                flex-direction: column;
                align-items: flex-start;
            }
            
            .action-buttons {
                width: 100%;
                justify-content: flex-start;
            }
            
            h1 {
                font-size: 24px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Spam Detection</h1>
            <div class="status-badge">
                <span class="dot"></span>
                AI Active
            </div>
        </div>
        <p class="subtitle">Real-time spam detection powered by machine learning</p>
        
        <textarea id="messageInput" placeholder="Enter your message here..."></textarea>
        
        <div class="button-group">
            <button class="btn btn-primary" onclick="analyzeMessage()">🔍 Analyze</button>
            <button class="btn btn-secondary" onclick="clearInput()">🗑️ Clear</button>
        </div>
        
        <div class="loading" id="loading">
            <span class="spinner"></span> Analyzing message...
        </div>
        
        <div id="result" class="result"></div>
        
        <div class="stats" id="stats">
            <div class="stat-card">
                <div class="number total-color" id="totalScans">0</div>
                <div class="label">📊 Total Scans</div>
            </div>
            <div class="stat-card">
                <div class="number spam-color" id="spamCount">0</div>
                <div class="label">🚫 Spam Detected</div>
            </div>
            <div class="stat-card">
                <div class="number legitimate-color" id="legitimateCount">0</div>
                <div class="label">✅ Legitimate</div>
            </div>
        </div>
        
        <div class="history-section">
            <div class="history-header">
                <h3>📜 Recent Scans</h3>
                <button class="refresh-btn" onclick="loadHistory()">🔄 Refresh</button>
            </div>
            <div class="history-container" id="historyContainer">
                <div id="history"></div>
            </div>
        </div>
    </div>

    <script>
        // Load stats and history on page load
        window.onload = function() {
            loadStats();
            loadHistory();
        };

        async function analyzeMessage() {
            const text = document.getElementById('messageInput').value.trim();
            if (!text) {
                showToast('Please enter a message to analyze.', 'error');
                return;
            }

            const loading = document.getElementById('loading');
            loading.classList.add('show');
            const resultDiv = document.getElementById('result');
            resultDiv.className = 'result';
            resultDiv.style.display = 'none';

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.error || 'Error analyzing message');
                }

                resultDiv.className = 'result ' + data.prediction;
                resultDiv.style.display = 'block';
                const confidencePercent = data.confidence;
                resultDiv.innerHTML = `
                    <h3>📊 Analysis Result</h3>
                    <div class="prediction">${data.prediction.toUpperCase()}</div>
                    <div class="confidence">Confidence: ${confidencePercent}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                    </div>
                    <div class="probabilities">
                        🚫 Spam: ${data.probabilities.spam}% &nbsp;|&nbsp; ✅ Legitimate: ${data.probabilities.legitimate}%
                    </div>
                    <div class="timestamp">
                        ID: #${data.id} | ${new Date(data.timestamp).toLocaleString()}
                    </div>
                `;

                // Refresh stats and history
                await loadStats();
                await loadHistory();
                showToast('✅ Analysis complete!', 'success');
                
            } catch (error) {
                showToast('Error: ' + error.message, 'error');
            } finally {
                loading.classList.remove('show');
            }
        }

        function clearInput() {
            document.getElementById('messageInput').value = '';
            document.getElementById('result').className = 'result';
            document.getElementById('result').style.display = 'none';
        }

        async function loadStats() {
            try {
                const response = await fetch('/stats');
                const data = await response.json();
                document.getElementById('totalScans').textContent = data.total || 0;
                document.getElementById('spamCount').textContent = data.spam || 0;
                document.getElementById('legitimateCount').textContent = data.legitimate || 0;
            } catch (error) {
                console.error('Error loading stats:', error);
            }
        }

        async function loadHistory() {
            try {
                const response = await fetch('/history?limit=10');
                if (!response.ok) {
                    throw new Error('Failed to load history');
                }
                const data = await response.json();
                
                const historyDiv = document.getElementById('history');
                if (!data || data.length === 0) {
                    historyDiv.innerHTML = '<div class="no-history">📭 No scans yet. Start analyzing messages!</div>';
                    return;
                }
                
                historyDiv.innerHTML = data.map(item => {
                    const confidencePercent = Math.round(item.confidence * 100);
                    return `
                    <div class="history-item" id="history-item-${item.id}">
                        <div class="message-text">
                            <span class="badge ${item.prediction}">${item.prediction}</span>
                            <span style="margin-left:10px;color:#aaa;">${item.message.substring(0, 60)}${item.message.length > 60 ? '...' : ''}</span>
                            <span style="margin-left:8px;font-size:11px;color:#555;">${confidencePercent}%</span>
                        </div>
                        <div class="action-buttons">
                            <span class="date">${new Date(item.created_at).toLocaleString()}</span>
                            <button class="btn btn-danger" onclick="deleteHistory(${item.id})">Delete</button>
                        </div>
                    </div>
                `}).join('');
            } catch (error) {
                console.error('Error loading history:', error);
                document.getElementById('history').innerHTML = '<div class="no-history">⚠️ Error loading history. Please refresh.</div>';
            }
        }

        async function deleteHistory(id) {
            if (!confirm('Delete this record?')) return;
            try {
                const response = await fetch('/history/' + id, { 
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    const data = await response.json();
                    if (data.success) {
                        const item = document.getElementById('history-item-' + id);
                        if (item) {
                            item.style.opacity = '0';
                            item.style.transition = 'opacity 0.3s';
                            setTimeout(() => {
                                loadHistory();
                                loadStats();
                            }, 300);
                        } else {
                            loadHistory();
                            loadStats();
                        }
                        showToast('✅ Record deleted successfully!', 'success');
                    } else {
                        showToast('Error: ' + data.error, 'error');
                    }
                } else {
                    const error = await response.json();
                    showToast('Error: ' + (error.error || 'Failed to delete'), 'error');
                }
            } catch (error) {
                showToast('Error deleting record: ' + error.message, 'error');
            }
        }

        // Toast notification
        function showToast(message, type = 'success') {
            const existingToast = document.querySelector('.toast');
            if (existingToast) {
                existingToast.remove();
            }
            
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.textContent = message;
            document.body.appendChild(toast);
            
            setTimeout(() => {
                toast.style.opacity = '0';
                toast.style.transition = 'opacity 0.3s';
                setTimeout(() => toast.remove(), 300);
            }, 3000);
        }
    </script>
</body>
</html>
"""

# ============================================================
# HTTP Request Handler
# ============================================================
class SpamDetectionHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        if path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
            
        elif path == '/stats':
            try:
                stats = db.get_stats()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(stats).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
            
        elif path == '/history':
            try:
                query = urllib.parse.parse_qs(parsed_path.query)
                limit = int(query.get('limit', [50])[0])
                history = db.get_history(limit)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(history, default=str).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
            
        elif path.startswith('/history/'):
            try:
                id_str = path.split('/')[2]
                if not id_str:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Missing ID"}).encode('utf-8'))
                    return
                    
                id = int(id_str)
                
                if db.delete_history(id):
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": True, "message": "Deleted successfully"}).encode('utf-8'))
                else:
                    self.send_response(404)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": "Record not found"}).encode('utf-8'))
            except ValueError as e:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": f"Invalid ID format: {str(e)}"}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Not Found')

    def do_POST(self):
        if self.path == '/analyze':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                text = data.get('text', '').strip()
                
                if not text:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Message cannot be empty"}).encode('utf-8'))
                    return
                
                result = spam_detector.predict(text)
                scan_id = db.save_scan(
                    text[:500],
                    result["prediction"],
                    result["confidence"] / 100
                )
                
                response = {
                    "id": scan_id,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"],
                    "timestamp": datetime.now().isoformat()
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')

    def do_DELETE(self):
        if self.path.startswith('/history/'):
            try:
                id_str = self.path.split('/')[2]
                if not id_str:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": "Missing ID"}).encode('utf-8'))
                    return
                    
                id = int(id_str)
                
                if db.delete_history(id):
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": True, "message": "Deleted successfully"}).encode('utf-8'))
                else:
                    self.send_response(404)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": "Record not found"}).encode('utf-8'))
            except ValueError as e:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": f"Invalid ID format: {str(e)}"}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        pass

# ============================================================
# Server
# ============================================================
def find_available_port(start_port=8000, max_port=8010):
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    return start_port

def run_server():
    port = find_available_port(PORT)
    
    print("="*60)
    print("🛡️ AI-Powered Spam Detection System")
    print("="*60)
    print(f"📍 Server: http://localhost:{port}")
    print(f"📊 Database: {DATABASE_PATH}")
    print(f"🤖 Model: {MODELS_DIR}/spam_model.pkl")
    print("="*60)
    print("✅ Server is running!")
    print("🌐 Opening browser...")
    print("Press Ctrl+C to stop")
    print("="*60)
    
    webbrowser.open(f"http://localhost:{port}")
    
    server = HTTPServer(('0.0.0.0', port), SpamDetectionHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        server.shutdown()

# ============================================================
# Main Entry Point
# ============================================================
if __name__ == "__main__":
    run_server()