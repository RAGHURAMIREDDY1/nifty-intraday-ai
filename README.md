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
## 🏗️ Architecture

\```
┌─────────────────────────────────────────────────┐
│              USER INPUT                          │
│         "Analyse Nifty Market"                  │
└──────────────────┬──────────────────────────────┘
                   ↓
┌──────────────────────────────────────────────────┐
│  AGENT 1 — Stock Scanner                         │
│  Scans all 50 Nifty stocks                       │
│  Scores by price movement + volume spike         │
│  Output: Top 5 most active stocks                │
└──────────────────┬───────────────────────────────┘
                   ↓
┌──────────────────────────────────────────────────┐
│  AGENT 2 — Strategy Expert                       │
│  RSI · VWAP · EMA 20/50 · Bollinger Bands        │
│  MACD · Opening Range Breakout · Support/Resist  │
│  Output: Confluence Score (0-100) per stock      │
└──────────────────┬───────────────────────────────┘
                   ↓
┌──────────────────────────────────────────────────┐
│  AGENT 3 — Scenario Tester (Gemini LLM)          │
│  Tests Bull / Bear / Sideways scenarios          │
│  Assigns opportunity score & risk factors        │
└──────────────────┬───────────────────────────────┘
                   ↓
┌──────────────────────────────────────────────────┐
│  AGENT 4 — Signal Generator (Gemini LLM)         │
│  BUY / SELL / AVOID signals                      │
│  Entry Price · Target · Stop Loss                │
│  🏆 TOP 2 highest confidence picks               │
└──────────────────────────────────────────────────┘
\```
## 🛠️ Tech Stack

- **LangGraph** — Multi-agent pipeline orchestration
- **Google Gemini** — LLM for scenario analysis and signal generation
- **yfinance** — Live NSE stock data (15-min delayed)
- **pandas-ta** — Technical indicators (RSI, MACD, Bollinger Bands, VWAP, EMA)
- **Gradio** — Web UI

## 🚀 Live Demo

[View on Hugging Face Spaces](https://huggingface.co/spaces/Raghu7207/Intradayanalyser)

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
## 📚 What I Learned Building This

- How to design and connect multiple AI agents using **LangGraph StateGraph**
- How to manage **shared state** across agents using TypedDict
- How to calculate **7 technical indicators** programmatically with pandas-ta
- How to handle **API rate limits** gracefully with auto model switching
- How to deploy AI apps on **Hugging Face Spaces**
- Best practices for **securing API keys** in cloud deployments
```

---

**4. Update commit messages** — right now your commits have no messages. Future commits should always have clear messages like:
```
✅ Good:  "Add Bollinger Bands fix for pandas-ta version compatibility"
❌ Bad:   "Update app.py"
```

---

**5. Pin this repo on your profile** — go to your profile page → "Customize your pins" → select `nifty-intraday-ai` as the first pin. This is the first thing recruiters see!

---

### 🎯 Overall verdict:
```
Current state    →  6/10 (good foundation)
After fixes      →  8.5/10 (very impressive for a beginner)
After Angel One  →  9.5/10 (standout portfolio project)
