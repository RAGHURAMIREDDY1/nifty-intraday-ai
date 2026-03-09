# 🇮🇳 Nifty 50 Intraday AI Analyst

A multi-agent AI system that analyses NSE Nifty 50 stocks for intraday trading opportunities using LangGraph and Google Gemini.

## 🤖 How it works

This app uses a 4-agent LangGraph pipeline:
```
Agent 1 — Stock Scanner     → Scans all 50 Nifty stocks, picks top 5 by activity
Agent 2 — Strategy Expert   → Calculates RSI, VWAP, EMA, Bollinger Bands, MACD, ORB, S&R
Agent 3 — Scenario Tester   → Tests Bull / Bear / Sideways scenarios using Gemini AI
Agent 4 — Signal Generator  → Generates BUY/SELL signals with Entry, Target & Stop Loss
```

## 🛠️ Tech Stack

- **LangGraph** — Multi-agent pipeline orchestration
- **Google Gemini** — LLM for scenario analysis and signal generation
- **yfinance** — Live NSE stock data (15-min delayed)
- **pandas-ta** — Technical indicators (RSI, MACD, Bollinger Bands, VWAP, EMA)
- **Gradio** — Web UI

## 🚀 Live Demo

[View on Hugging Face Spaces](your-huggingface-link-here)

## ⚙️ Setup

1. Clone the repo
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set your Gemini API key:
```bash
export GOOGLE_API_KEY=your_key_here
```
4. Run:
```bash
python app.py
```

## 📊 Features

- Scans all 50 Nifty stocks automatically
- 7 technical indicators with confluence scoring (0-100)
- Auto model switching if API rate limit is hit
- Step-by-step agent output in real time
- Top 2 highest confidence picks highlighted

## ⚠️ Disclaimer

This tool is for **educational purposes only**.
Not financial advice. Always consult a SEBI-registered advisor before trading.

## 👨‍💻 Author

Built as part of learning AI Engineering with LangGraph and multi-agent systems.
