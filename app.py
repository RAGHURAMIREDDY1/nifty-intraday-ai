import os
import time
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import gradio as gr
from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# ── API Key ─────────────────────────────────────────────────────────────────
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
if not os.environ["GOOGLE_API_KEY"]:
    raise ValueError("❌ GOOGLE_API_KEY secret is missing! Add it in Space Settings → Secrets.")

# ── LLM Model Manager with rate-limit auto-switching ────────────────────────
GEMINI_MODELS       = [
    "gemini-3-flash",          # Primary   — try this first
    "gemini-2.5-flash",        # Fallback  — switch if Gemini 3 hits rate limit
]
RATE_LIMIT_KEYWORDS = ["429", "quota", "rate", "RESOURCE_EXHAUSTED", "limit"]
NOT_FOUND_KEYWORDS  = ["404", "NOT_FOUND", "not found", "not supported"]

class LLMManager:
    """
    Manages multiple Gemini models with auto-switching:
    - Startup    : picks first working model
    - Rate limit : switches to next model automatically
    - 404 error  : switches to next model automatically
    - All gone   : raises clear user-friendly error
    """
    def __init__(self, models):
        self.models        = models
        self.llm           = None
        self.loaded_model  = None
        self.exhausted     = []
        self._init_first()

    def _init_first(self):
        for model_name in self.models:
            try:
                print(f"⏳ Trying: {model_name}")
                candidate = ChatGoogleGenerativeAI(model=model_name)
                candidate.invoke("say hi in 2 words")
                self.llm          = candidate
                self.loaded_model = model_name
                print(f"✅ Active model: {model_name}")
                return
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
                self.exhausted.append(model_name)
        raise RuntimeError("❌ All Gemini models failed. Check your GOOGLE_API_KEY in Space Secrets.")

    def _is_rate_limit(self, e):
        return any(k.lower() in str(e).lower() for k in RATE_LIMIT_KEYWORDS)

    def _is_not_found(self, e):
        return any(k.lower() in str(e).lower() for k in NOT_FOUND_KEYWORDS)

    def switch_next(self, reason=""):
        self.exhausted.append(self.loaded_model)
        remaining = [m for m in self.models if m not in self.exhausted]
        if not remaining:
            raise RuntimeError(
                f"❌ ALL GEMINI MODELS EXHAUSTED\n"
                f"Tried: {', '.join(self.exhausted)}\n"
                f"Reason: {reason}\n"
                f"💡 Free tier quota resets every hour. Please wait and try again."
            )
        for model_name in remaining:
            try:
                print(f"🔄 Switching → {model_name}  (reason: {reason})")
                candidate = ChatGoogleGenerativeAI(model=model_name)
                candidate.invoke("say hi in 2 words")
                self.llm          = candidate
                self.loaded_model = model_name
                print(f"✅ Now using: {model_name}")
                return
            except Exception as e:
                print(f"❌ {model_name} also failed: {e}")
                self.exhausted.append(model_name)
        raise RuntimeError(
            "❌ ALL GEMINI MODELS EXHAUSTED\n"
            "💡 Free tier quota resets every hour. Please wait and try again."
        )

    def invoke(self, prompt):
        return self.llm.invoke(prompt)

llm_manager  = LLMManager(GEMINI_MODELS)
loaded_model = llm_manager.loaded_model

# ── Nifty 50 ─────────────────────────────────────────────────────────────────
NIFTY50 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS","ITC.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS",
    "TITAN.NS","ULTRACEMCO.NS","BAJFINANCE.NS","WIPRO.NS","NESTLEIND.NS",
    "POWERGRID.NS","NTPC.NS","TECHM.NS","HCLTECH.NS","ONGC.NS",
    "TATAMOTORS.NS","ADANIENT.NS","JSWSTEEL.NS","TATASTEEL.NS","HINDALCO.NS",
    "INDUSINDBK.NS","BAJAJFINSV.NS","BPCL.NS","DIVISLAB.NS","DRREDDY.NS",
    "EICHERMOT.NS","GRASIM.NS","HEROMOTOCO.NS","M&M.NS","CIPLA.NS",
    "COALINDIA.NS","BRITANNIA.NS","APOLLOHOSP.NS","TATACONSUM.NS","ADANIPORTS.NS",
    "SBILIFE.NS","HDFCLIFE.NS","UPL.NS","BAJAJ-AUTO.NS","VEDL.NS"
]

# ── Shared State ──────────────────────────────────────────────────────────────
class MarketState(TypedDict):
    top_stocks: List[str]
    indicators: dict
    scenario_analysis: str
    signals: str
    errors: List[str]

# ── Retry helper ──────────────────────────────────────────────────────────────
def extract_text_from_response(content) -> str:
    """
    Handles all Gemini response formats:
    - Gemini 2.x → content is a plain string
    - Gemini 3.x → content is a list of dicts: [{'type': 'text', 'text': '...', 'extras': {...}}]
    """
    # Already a plain string (Gemini 2.x)
    if isinstance(content, str):
        return content.strip()

    # List of content blocks (Gemini 3.x)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()

    # Fallback — convert whatever it is to string
    return str(content).strip()

def safe_llm_invoke(prompt: str, retries: int = 3, delay: int = 5) -> str:
    """
    Calls LLM with:
    - Auto-switch on rate limit (429) or model not found (404)
    - Retry with delay on transient errors
    - Clear error message if all models exhausted
    """
    global loaded_model
    for attempt in range(retries):
        try:
            response = llm_manager.invoke(prompt)
            loaded_model = llm_manager.loaded_model  # keep in sync for UI
            return extract_text_from_response(response.content)

        except Exception as e:
            err_str = str(e)
            # ── Rate limit or model not found → switch immediately ──
            if llm_manager._is_rate_limit(e) or llm_manager._is_not_found(e):
                reason = "rate limit (429)" if llm_manager._is_rate_limit(e) else "model not found (404)"
                print(f"⚠️ {reason} on {llm_manager.loaded_model} — switching model...")
                try:
                    llm_manager.switch_next(reason=reason)
                    loaded_model = llm_manager.loaded_model
                    print(f"🔄 Retrying with {loaded_model}...")
                    continue  # retry immediately with new model
                except RuntimeError as switch_err:
                    return f"❌ {switch_err}"

            # ── Transient error → wait and retry ──
            print(f"⚠️ Attempt {attempt+1}/{retries} failed: {err_str[:100]}")
            if attempt < retries - 1:
                time.sleep(delay)

    return "⚠️ LLM call failed after all retries. Please try again in a few minutes."

# ════════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Stock Scanner
# Scores all 50 stocks by price movement + volume spike → picks top 5
# ════════════════════════════════════════════════════════════════════════════════
def stock_scanner_agent(state: MarketState) -> MarketState:
    print("🔍 Agent 1: Scanning Nifty 50...")
    scored_stocks = []
    errors = []

    for symbol in NIFTY50:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(symbol, period="5d", interval="1h",
                                 progress=False, auto_adjust=True)
            if df is None or df.empty or len(df) < 5:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            latest  = df.iloc[-1]
            prev    = df.iloc[-2]
            price_change  = abs((float(latest["Close"]) - float(prev["Close"])) / float(prev["Close"])) * 100
            avg_volume    = float(df["Volume"].mean())
            volume_score  = float(latest["Volume"]) / avg_volume if avg_volume > 0 else 0
            score         = price_change + volume_score
            scored_stocks.append((symbol, round(score, 3)))
        except Exception as e:
            errors.append(f"Scanner skipped {symbol}: {e}")
            continue

    scored_stocks.sort(key=lambda x: x[1], reverse=True)
    state["top_stocks"] = [s[0] for s in scored_stocks[:5]]
    state["errors"]     = errors
    print(f"✅ Top 5: {state['top_stocks']}")
    return state

# ════════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Strategy Expert
# Calculates all 7 indicators + confluence score for each stock
# ════════════════════════════════════════════════════════════════════════════════
def strategy_expert_agent(state: MarketState) -> MarketState:
    print("📊 Agent 2: Calculating 7 indicators...")
    all_indicators = {}

    for symbol in state["top_stocks"]:
        try:
            df = yf.download(symbol, period="10d", interval="1h", progress=False)
            if df.empty or len(df) < 30:
                state["errors"].append(f"Strategy: not enough data for {symbol}")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close  = df["Close"].squeeze()
            high   = df["High"].squeeze()
            low    = df["Low"].squeeze()
            volume = df["Volume"].squeeze()
            open_  = df["Open"].squeeze()

            current_price = round(float(close.iloc[-1]), 2)
            signals_score = 0   # confluence score out of 100
            signal_tags   = []  # which indicators are bullish

            # ── 1. RSI (weight: 15) ──────────────────────────────────────────
            # <30 oversold=BUY signal, >70 overbought=SELL signal
            rsi_series = ta.rsi(close, length=14)
            rsi_val    = round(float(rsi_series.iloc[-1]), 2) if rsi_series is not None else 50
            if rsi_val < 35:
                signals_score += 15
                signal_tags.append("RSI Oversold ✅")
            elif rsi_val > 65:
                signals_score -= 10
                signal_tags.append("RSI Overbought ⚠️")
            else:
                signals_score += 8
                signal_tags.append("RSI Neutral")

            # ── 2. VWAP (weight: 20) ─────────────────────────────────────────
            # Price above VWAP = bullish intraday sentiment
            vwap_series = ta.vwap(high, low, close, volume)
            vwap_val    = round(float(vwap_series.iloc[-1]), 2) if vwap_series is not None else current_price
            if current_price > vwap_val:
                signals_score += 20
                signal_tags.append("Above VWAP ✅")
            else:
                signals_score -= 5
                signal_tags.append("Below VWAP ⚠️")

            # ── 3. EMA Crossover 20/50 (weight: 20) ──────────────────────────
            # EMA20 > EMA50 = bullish trend
            ema20_series = ta.ema(close, length=20)
            ema50_series = ta.ema(close, length=50)
            ema20_val    = round(float(ema20_series.iloc[-1]), 2) if ema20_series is not None else current_price
            ema50_val    = round(float(ema50_series.iloc[-1]), 2) if ema50_series is not None else current_price
            if ema20_val > ema50_val and current_price > ema20_val:
                signals_score += 20
                signal_tags.append("EMA Bullish Cross ✅")
            elif ema20_val < ema50_val:
                signals_score -= 10
                signal_tags.append("EMA Bearish Cross ⚠️")
            else:
                signals_score += 5
                signal_tags.append("EMA Neutral")

            # ── 4. Bollinger Bands (weight: 15) ──────────────────────────────
            # Price near lower band = potential bounce (BUY)
            # Price near upper band = potential reversal (SELL)
            bb         = ta.bbands(close, length=20, std=2)
            bb_upper   = current_price
            bb_lower   = current_price
            bb_mid     = current_price
            if bb is not None:
                # column names vary by pandas-ta version — find them dynamically
                upper_col = [c for c in bb.columns if "BBU" in c]
                lower_col = [c for c in bb.columns if "BBL" in c]
                mid_col   = [c for c in bb.columns if "BBM" in c]
                if upper_col: bb_upper = round(float(bb[upper_col[0]].iloc[-1]), 2)
                if lower_col: bb_lower = round(float(bb[lower_col[0]].iloc[-1]), 2)
                if mid_col:   bb_mid   = round(float(bb[mid_col[0]].iloc[-1]),   2)
            bb_width = round(((bb_upper - bb_lower) / bb_mid) * 100, 2) if bb_mid > 0 else 0
            if current_price <= bb_lower * 1.01:
                signals_score += 15
                signal_tags.append("BB Lower Bounce ✅")
            elif current_price >= bb_upper * 0.99:
                signals_score -= 10
                signal_tags.append("BB Upper Resistance ⚠️")
            else:
                signals_score += 5
                signal_tags.append("BB Mid Zone")

            # ── 5. MACD (weight: 15) ─────────────────────────────────────────
            # MACD line crossing above signal line = BUY
            macd_df      = ta.macd(close, fast=12, slow=26, signal=9)
            macd_val     = round(float(macd_df["MACD_12_26_9"].iloc[-1]), 4)   if macd_df is not None else 0
            macd_signal  = round(float(macd_df["MACDs_12_26_9"].iloc[-1]), 4)  if macd_df is not None else 0
            macd_hist    = round(float(macd_df["MACDh_12_26_9"].iloc[-1]), 4)  if macd_df is not None else 0
            if macd_val > macd_signal and macd_hist > 0:
                signals_score += 15
                signal_tags.append("MACD Bullish ✅")
            elif macd_val < macd_signal:
                signals_score -= 8
                signal_tags.append("MACD Bearish ⚠️")
            else:
                signal_tags.append("MACD Neutral")

            # ── 6. Opening Range Breakout (weight: 10) ───────────────────────
            # First 15-min high/low = ORB zone
            # Price breaking above ORB high = strong BUY signal
            orb_df    = yf.download(symbol, period="1d", interval="15m", progress=False)
            orb_score = 0
            orb_tag   = "ORB: No data"
            if not orb_df.empty and len(orb_df) >= 2:
                if isinstance(orb_df.columns, pd.MultiIndex):
                    orb_df.columns = orb_df.columns.get_level_values(0)
                orb_high = round(float(orb_df["High"].iloc[0]), 2)
                orb_low  = round(float(orb_df["Low"].iloc[0]), 2)
                if current_price > orb_high:
                    signals_score += 10
                    orb_tag = f"ORB Breakout ✅ (H:{orb_high})"
                elif current_price < orb_low:
                    signals_score -= 5
                    orb_tag = f"ORB Breakdown ⚠️ (L:{orb_low})"
                else:
                    orb_tag = f"ORB Inside (H:{orb_high} L:{orb_low})"
            signal_tags.append(orb_tag)

            # ── 7. Support & Resistance (weight: 5) ──────────────────────────
            # Recent swing high = resistance, swing low = support
            recent_high = round(float(high.tail(20).max()), 2)
            recent_low  = round(float(low.tail(20).min()), 2)
            near_support    = current_price <= recent_low * 1.02
            near_resistance = current_price >= recent_high * 0.98
            if near_support:
                signals_score += 5
                signal_tags.append(f"Near Support ✅ (₹{recent_low})")
            elif near_resistance:
                signals_score -= 5
                signal_tags.append(f"Near Resistance ⚠️ (₹{recent_high})")
            else:
                signal_tags.append(f"S/R: Mid zone (S:₹{recent_low} R:₹{recent_high})")

            # ── Final confluence decision ─────────────────────────────────────
            signals_score = max(0, min(100, signals_score))
            if signals_score >= 65:
                confluence = "STRONG BUY"
            elif signals_score >= 50:
                confluence = "BUY"
            elif signals_score >= 35:
                confluence = "NEUTRAL"
            else:
                confluence = "AVOID"

            # ── Stop loss based on swing low (not fixed %) ───────────────────
            swing_low_sl  = round(recent_low, 2)
            pct_sl        = round(current_price * 0.985, 2)
            stop_loss     = max(swing_low_sl, pct_sl)  # tighter of the two

            all_indicators[symbol] = {
                "current_price"  : current_price,
                "rsi"            : rsi_val,
                "vwap"           : vwap_val,
                "ema20"          : ema20_val,
                "ema50"          : ema50_val,
                "bb_upper"       : bb_upper,
                "bb_lower"       : bb_lower,
                "bb_width"       : bb_width,
                "macd"           : macd_val,
                "macd_signal"    : macd_signal,
                "macd_hist"      : macd_hist,
                "support"        : recent_low,
                "resistance"     : recent_high,
                "confluence_score": signals_score,
                "confluence"     : confluence,
                "stop_loss"      : stop_loss,
                "signal_tags"    : signal_tags,
                "price_vs_vwap"  : "ABOVE" if current_price > vwap_val else "BELOW",
                "trend"          : "BULLISH" if ema20_val > ema50_val else "BEARISH",
            }
            print(f"  ✅ {symbol}: ₹{current_price} | Score:{signals_score} | {confluence}")

        except Exception as e:
            state["errors"].append(f"Strategy failed for {symbol}: {e}")
            print(f"  ❌ {symbol}: {e}")
            continue

    state["indicators"] = all_indicators
    return state

# ════════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Scenario Tester
# Sends all indicator data to Gemini — tests Bull/Bear/Sideways
# ════════════════════════════════════════════════════════════════════════════════
def scenario_tester_agent(state: MarketState) -> MarketState:
    print("🧪 Agent 3: Testing scenarios...")

    indicators_text = ""
    for symbol, d in state["indicators"].items():
        indicators_text += f"""
━━━ {symbol.replace('.NS','')} ━━━
Price: ₹{d['current_price']} | Trend: {d['trend']} | Confluence: {d['confluence']} ({d['confluence_score']}/100)
RSI: {d['rsi']} | VWAP: ₹{d['vwap']} ({d['price_vs_vwap']})
EMA20: ₹{d['ema20']} | EMA50: ₹{d['ema50']}
BB Upper: ₹{d['bb_upper']} | BB Lower: ₹{d['bb_lower']} | BB Width: {d['bb_width']}%
MACD: {d['macd']} | Signal: {d['macd_signal']} | Hist: {d['macd_hist']}
Support: ₹{d['support']} | Resistance: ₹{d['resistance']}
Active Signals: {', '.join(d['signal_tags'])}
"""

    prompt = f"""You are a senior Indian stock market intraday analyst with 15+ years NSE experience.
Analyze these Nifty 50 stocks using their technical indicators.
For EACH stock test 3 scenarios:
1. 🐂 BULLISH — conditions for upside move
2. 🐻 BEARISH — conditions for downside risk  
3. ↔️ SIDEWAYS — consolidation scenario
Stock Data:
{indicators_text}
For each stock provide:
• Scenario probabilities (Bull X% / Bear Y% / Sideways Z%)
• Key price levels to watch
• Opportunity Score: X/10
• Best strategy to use for this stock today (from: ORB, VWAP Reversion, EMA Crossover, BB Squeeze, MACD Momentum)
• One key risk factor
Rules:
- Use Indian market context (NSE, 9:15AM–3:30PM IST)
- Consider global cues, FII/DII activity in your reasoning
- Be specific with price levels in ₹
- Keep each stock analysis concise (5-6 lines max)"""

    state["scenario_analysis"] = safe_llm_invoke(prompt)
    print("✅ Scenarios done!")
    return state

# ════════════════════════════════════════════════════════════════════════════════
# AGENT 4 — Signal Generator
# Generates BUY/SELL signals → picks TOP 2 by confidence
# ════════════════════════════════════════════════════════════════════════════════
def signal_generator_agent(state: MarketState) -> MarketState:
    print("🎯 Agent 4: Generating signals...")

    indicators_text = ""
    for symbol, d in state["indicators"].items():
        indicators_text += (
            f"{symbol.replace('.NS','')}: ₹{d['current_price']} | "
            f"Score:{d['confluence_score']}/100 | {d['confluence']} | "
            f"RSI:{d['rsi']} | Trend:{d['trend']} | SL:₹{d['stop_loss']} | "
            f"R:₹{d['resistance']} | S:₹{d['support']}\n"
        )

    prompt = f"""You are a professional intraday trading advisor for NSE Indian markets.
Generate precise trading signals for each stock below, then pick the TOP 2 with highest confidence.
Technical Summary:
{indicators_text}
Scenario Analysis:
{state['scenario_analysis']}
Generate for EACH stock in EXACTLY this format (no extra text):
STOCK: [NAME]
SIGNAL: [BUY / SELL / AVOID]
ENTRY: ₹[price]
TARGET 1: ₹[price]  (+X%)
TARGET 2: ₹[price]  (+X%)
STOP LOSS: ₹[price]  (-X%)
RISK-REWARD: 1:[ratio]
CONFIDENCE: [X]%
STRATEGY: [which strategy triggered this]
REASON: [max 15 words]
----------
After ALL stocks, add:
═══════════════════════════
🏆 TOP 2 HIGH CONFIDENCE PICKS
═══════════════════════════
🥇 PICK 1: [NAME] | [SIGNAL] | CONFIDENCE: [X]% | Entry: ₹[price] | Target: ₹[price] | SL: ₹[price]
🥈 PICK 2: [NAME] | [SIGNAL] | CONFIDENCE: [X]% | Entry: ₹[price] | Target: ₹[price] | SL: ₹[price]
Strict rules:
- Stop loss MAX 1.5% from entry for intraday
- Minimum 1:2 risk-reward ratio always
- AVOID if confluence score < 40
- All prices in ₹ (Indian Rupees)
- Intraday only — exit before 3:15 PM IST
- Entry price must be within 0.3% of current price
⚠️ EDUCATIONAL PURPOSE ONLY — NOT FINANCIAL ADVICE"""

    state["signals"] = safe_llm_invoke(prompt)
    print("✅ Signals done!")
    return state

# ── Build LangGraph ───────────────────────────────────────────────────────────
graph = StateGraph(MarketState)
graph.add_node("scanner",   stock_scanner_agent)
graph.add_node("strategy",  strategy_expert_agent)
graph.add_node("scenarios", scenario_tester_agent)
graph.add_node("signals",   signal_generator_agent)
graph.set_entry_point("scanner")
graph.add_edge("scanner",   "strategy")
graph.add_edge("strategy",  "scenarios")
graph.add_edge("scenarios", "signals")
graph.add_edge("signals",   END)
pipeline = graph.compile()

# ════════════════════════════════════════════════════════════════════════════════
# GRADIO UI — Dark Terminal Theme
# ════════════════════════════════════════════════════════════════════════════════
def format_indicators(indicators: dict) -> str:
    if not indicators:
        return "No data yet..."
    lines = []
    for symbol, d in indicators.items():
        score     = d['confluence_score']
        bar_filled = int(score / 10)
        bar        = "█" * bar_filled + "░" * (10 - bar_filled)
        lines.append(f"""
┌─ {symbol.replace('.NS','')} {'─'*(20-len(symbol))}┐
│ Price    : ₹{d['current_price']}
│ Score    : [{bar}] {score}/100 → {d['confluence']}
│ RSI      : {d['rsi']}  {'🔴 Overbought' if d['rsi']>70 else '🟢 Oversold' if d['rsi']<30 else '🟡 Neutral'}
│ VWAP     : ₹{d['vwap']}  ({d['price_vs_vwap']})
│ EMA20/50 : ₹{d['ema20']} / ₹{d['ema50']}  {d['trend']}
│ BB Band  : ₹{d['bb_lower']} ─── ₹{d['bb_upper']}
│ MACD     : {d['macd']}  Hist: {d['macd_hist']}
│ Support  : ₹{d['support']}  Resistance: ₹{d['resistance']}
│ Stop Loss: ₹{d['stop_loss']}
└{'─'*32}┘""")
    return "\n".join(lines)

def run_analysis():
    state = {
        "top_stocks": [],
        "indicators": {},
        "scenario_analysis": "",
        "signals": "",
        "errors": []
    }

    # ── Agent 1 ──
    yield ("⏳ Agent 1 scanning all 50 Nifty stocks...", "", "", "", f"🔄 Agent 1 running | Model: {llm_manager.loaded_model}")
    state = stock_scanner_agent(state)
    if not state["top_stocks"]:
        yield ("❌ No stocks found. Market may be closed or data unavailable.", "", "", "", "❌ Failed")
        return
    stocks_text = f"✅ Top 5 Most Active Stocks Selected:\n\n"
    for i, s in enumerate(state["top_stocks"], 1):
        stocks_text += f"  {i}. {s.replace('.NS','')}\n"
    stocks_text += f"\n📊 Scoring Method: Price Movement % + Volume Spike Ratio"
    yield (stocks_text, "⏳ Agent 2 calculating 7 indicators...", "", "", f"🔄 Agent 2 running | Model: {llm_manager.loaded_model}")

    # ── Agent 2 ──
    state = strategy_expert_agent(state)
    if not state["indicators"]:
        yield (stocks_text, "❌ Could not calculate indicators.", "", "", "❌ Failed")
        return
    indicators_text = format_indicators(state["indicators"])
    yield (stocks_text, indicators_text, "⏳ Agent 3 testing Bull/Bear/Sideways scenarios...", "", f"🔄 Agent 3 running | Model: {llm_manager.loaded_model}")

    # ── Agent 3 ──
    state = scenario_tester_agent(state)
    yield (stocks_text, indicators_text, state["scenario_analysis"], "⏳ Agent 4 generating final signals & Top 2 picks...", f"🔄 Agent 4 running | Model: {llm_manager.loaded_model}")

    # ── Agent 4 ──
    state = signal_generator_agent(state)
    error_text = ""
    if state["errors"]:
        error_text = f"\n\n⚠️ Non-critical warnings ({len(state['errors'])}):\n" + "\n".join(state["errors"][:3])

    yield (
        stocks_text,
        indicators_text,
        state["scenario_analysis"],
        state["signals"] + error_text,
        f"✅ Analysis Complete! | Active Model: {llm_manager.loaded_model} | Tried: {len(llm_manager.exhausted)} fallback(s)"
    )

# ── Dark CSS ──────────────────────────────────────────────────────────────────
dark_css = """
* { box-sizing: border-box; }
body, .gradio-container {
    background: #0a0e1a !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}
.gr-button-primary {
    background: linear-gradient(135deg, #00d4aa, #0099ff) !important;
    border: none !important;
    color: #0a0e1a !important;
    font-weight: 800 !important;
    font-size: 16px !important;
    letter-spacing: 1px !important;
    border-radius: 6px !important;
}
.gr-button-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 20px #00d4aa66 !important;
}
textarea, .gr-textbox textarea {
    background: #0d1117 !important;
    color: #00ff88 !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-size: 12px !important;
    border: 1px solid #1e2d40 !important;
    border-radius: 6px !important;
    line-height: 1.7 !important;
}
.gr-textbox label {
    color: #00d4aa !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}
.gr-markdown h1 { color: #00d4aa !important; font-size: 22px !important; }
.gr-markdown h3 { color: #0099ff !important; font-size: 13px !important; letter-spacing: 2px; }
.gr-markdown p  { color: #64748b !important; font-size: 12px !important; }
.gr-row { gap: 12px !important; }
"""

with gr.Blocks(
    title="Nifty 50 Intraday AI Terminal",
    css=dark_css
) as demo:

    gr.Markdown(f"""
# ⚡ NIFTY 50 INTRADAY AI TERMINAL
### 4-AGENT LANGGRAPH PIPELINE · LIVE NSE DATA · {loaded_model.upper()}
> ⚠️ **DISCLAIMER:** Educational & research purposes only. Not financial advice. Past performance does not guarantee future results. Always consult a SEBI-registered advisor before trading.
""")

    with gr.Row():
        analyse_btn = gr.Button(
            "▶  RUN MARKET ANALYSIS",
            variant="primary",
            size="lg"
        )

    status_box = gr.Textbox(
        label="◈ SYSTEM STATUS",
        interactive=False,
        lines=1,
        value="Ready. Click RUN to start analysis."
    )

    with gr.Row():
        agent1_box = gr.Textbox(
            label="◈ AGENT 1 — STOCK SCANNER",
            interactive=False,
            lines=10
        )
        agent2_box = gr.Textbox(
            label="◈ AGENT 2 — STRATEGY EXPERT  [RSI · VWAP · EMA · BB · MACD · ORB · S/R]",
            interactive=False,
            lines=10
        )

    agent3_box = gr.Textbox(
        label="◈ AGENT 3 — SCENARIO TESTER  [BULL · BEAR · SIDEWAYS]",
        interactive=False,
        lines=14
    )

    agent4_box = gr.Textbox(
        label="◈ AGENT 4 — SIGNAL GENERATOR  [TOP 2 HIGH CONFIDENCE PICKS]",
        interactive=False,
        lines=22
    )

    analyse_btn.click(
        fn=run_analysis,
        inputs=[],
        outputs=[agent1_box, agent2_box, agent3_box, agent4_box, status_box]
    )

demo.launch()
