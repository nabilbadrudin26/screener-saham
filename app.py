import os
import json
import joblib
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, date
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

# ==========================================
# 1. CLASS DEFINITION (REQUIRED FOR JOBLIB LOAD)
# ==========================================
class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.classes_ = np.array([0, 1])
        
        self.model_hgb = HistGradientBoostingClassifier(
            class_weight='balanced', l2_regularization=1.5, min_samples_leaf=30, random_state=self.random_state
        )
        self.model_rf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=7, min_samples_leaf=20, class_weight='balanced', random_state=self.random_state
        )

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_hgb.fit(X, y)
        self.model_rf.fit(X, y)
        return self

    def predict_proba(self, X):
        return (self.model_hgb.predict_proba(X) + self.model_rf.predict_proba(X)) / 2.0

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.520).astype(int)

# ==========================================
# 2. CONFIG PAGE & CUSTOM STYLES
# ==========================================
st.set_page_config(page_title="IDX AI Quant Scanner & Portfolio Tracker", layout="wide")

st.markdown("""
    <style>
    .reportview-container { background: #f8fafc; }
    .main-header {
        background-color: #012981;
        padding: 20px;
        border-radius: 6px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
    }
    .recom-card {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 8px;
        border-top: 4px solid #10b981;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <h2 style='margin:0; font-weight:800;'>IDX AI QUANT SCANNER & LIFECYCLE TRACKER</h2>
        <p style='margin:5px 0 0 0; opacity:0.8; font-size:13px;'>INSTITUTIONAL ENSEMBLE ENGINE V28.0 | TRIPLE BARRIER TIME-BOUND SYSTEM</p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 3. HELPER FUNCTIONS & PORTFOLIO PERSISTENCE
# ==========================================
# Mengambil konfigurasi dari secrets atau environment
try:
    BIN_ID = st.secrets["JSONBIN_BIN_ID"]
    API_KEY = st.secrets["JSONBIN_API_KEY"]
except KeyError:
    st.error("⚠️ Kredensial JSONBin belum diatur di Streamlit Secrets!")
    st.stop()

BASE_URL = f"https://api.jsonbin.io/v3/b/{BIN_ID}"
HEADERS = {
    "X-Master-Key": API_KEY,
    "Content-Type": "application/json"
}

def load_portfolio():
    """Memuat data portofolio dari cloud storage JSONBin.io"""
    try:
        response = requests.get(f"{BASE_URL}/latest", headers=HEADERS)
        if response.status_code == 200:
            # Data tersimpan di dalam field 'record'
            return response.json().get("record", [])
        else:
            st.error(f"Gagal memuat portofolio (Status Code: {response.status_code})")
            return []
    except Exception as e:
        st.error(f"Terjadi kesalahan koneksi saat memuat portofolio: {e}")
        return []

def save_portfolio(data):
    """Menyimpan data portofolio terbaru ke cloud storage JSONBin.io"""
    try:
        # PUT request untuk memperbarui isi JSON di JSONBin
        response = requests.put(BASE_URL, headers=HEADERS, json=data)
        if response.status_code == 200:
            st.success("Portofolio berhasil diperbarui ke Cloud!")
            return True
        else:
            st.error(f"Gagal menyimpan portofolio (Status Code: {response.status_code})")
            return False
    except Exception as e:
        st.error(f"Terjadi kesalahan koneksi saat menyimpan portofolio: {e}")
        return False

def calculate_target_date(start_date_str, trading_days=10):
    start_dt = np.datetime64(start_date_str)
    target_dt = np.busday_offset(start_dt, trading_days, roll='forward')
    return pd.to_datetime(str(target_dt)).strftime('%Y-%m-%d')

def count_business_days_elapsed(start_date_str):
    start_dt = np.datetime64(start_date_str)
    today_dt = np.datetime64(date.today())
    if today_dt <= start_dt:
        return 0
    return int(np.busday_count(start_dt, today_dt))

@st.cache_resource
def load_ml_assets():
    if not os.path.exists('model_v28.joblib') or not os.path.exists('metadata_v28.json'):
        return None, None
    model = joblib.load('model_v28.joblib')
    with open('metadata_v28.json', 'r') as f:
        metadata = json.load(f)
    return model, metadata

model, metadata = load_ml_assets()

if model is None:
    st.error("⚠️ File `model_v28.joblib` atau `metadata_v28.json` tidak ditemukan! Jalankan `train_and_save.py` terlebih dahulu.")
    st.stop()

# ==========================================
# 4. INDICATOR CALCULATOR
# ==========================================
def calculate_indicators(df):
    df = df.copy()
    high_low = df['High'] - df['Low']
    high_cp = np.abs(df['High'] - df['Close'].shift(1))
    low_cp = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['ATR_Pct'] = df['ATR'] / (df['Close'] + 1e-9)
    
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['Dist_SMA50'] = (df['Close'] - df['SMA_50']) / (df['SMA_50'] + 1e-9)
    df['Dist_SMA200'] = (df['Close'] - df['SMA_200']) / (df['SMA_200'] + 1e-9)
    
    df['High_20'] = df['High'].rolling(20).max()
    df['High_20_Ratio'] = (df['Close'] - df['High_20']) / (df['High_20'] + 1e-9)
    df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
    df['Vol_STD20'] = df['Volume'].rolling(20).std()
    df['Vol_ZScore'] = (df['Volume'] - df['Vol_SMA20']) / (df['Vol_STD20'] + 1e-9)
    
    df['Turnover_20MA'] = (df['Close'] * df['Volume']).rolling(20).mean()
    return df

# ==========================================
# 5. SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("### 🛰️ CONTROLS & MODEL ENGINE")
st.sidebar.success("✅ Model V28.0 Ready (Pre-Trained)")

tickers_basket = metadata['basket']
st.sidebar.markdown(f"**Total Semesta Saham:** `{len(tickers_basket)} Emiten`")

conf_threshold = st.sidebar.slider(
    "🎯 Minimal Keyakinan AI (%):", 
    min_value=50.0, max_value=70.0, 
    value=float(metadata['prob_threshold'] * 100), 
    step=0.5
) / 100.0

min_turnover = st.sidebar.select_slider(
    "🛡️ Filter Min. Turnover Harian:",
    options=[100_000_000, 300_000_000, 500_000_000, 750_000_000, 1_000_000_000],
    value=300_000_000,
    format_func=lambda x: f"Rp {x/1e6:,.0f} Juta"
)

st.sidebar.markdown("---")
tombol_scan = st.sidebar.button("📡 JALANKAN RADAR INSTAN", use_container_width=True)

# ==========================================
# 6. NAVIGATION TABS
# ==========================================
tab_radar, tab_portfolio = st.tabs(["📡 Screener Radar Saham", "💼 Tracker Portofolio Aktif"])

# ------------------------------------------
# TAB 1: SCREENER RADAR
# ------------------------------------------
with tab_radar:
    if tombol_scan:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.markdown("⏳ **Mengunduh Data Pasar Terbaru Secara Paralel...**")
        
        batch_size = 60
        all_results = []
        feature_cols = metadata['feature_cols']
        today_str = date.today().strftime('%Y-%m-%d')
        target_exit_str = calculate_target_date(today_str, trading_days=10)
        
        for idx in range(0, len(tickers_basket), batch_size):
            batch = tickers_basket[idx:idx + batch_size]
            pct = int(((idx + len(batch)) / len(tickers_basket)) * 100)
            progress_bar.progress(pct)
            status_text.markdown(f"**Memindai Batch Saham ({idx + 1} - {min(idx + batch_size, len(tickers_basket))}/{len(tickers_basket)})...**")
            
            try:
                data = yf.download(batch, period="1y", group_by='ticker', progress=False, threads=True)
                for t in batch:
                    try:
                        df = data[t].dropna(how='all') if len(batch) > 1 else data.dropna(how='all')
                        if df.empty or len(df) < 50:
                            continue
                        
                        df = calculate_indicators(df)
                        latest = df.iloc[-1].copy()
                        
                        if np.isnan(latest['Close']) or latest['Close'] < 50:
                            continue
                        
                        turnover = latest['Turnover_20MA']
                        X_input = pd.DataFrame([latest[feature_cols]])
                        
                        if X_input.isnull().values.any():
                            continue
                            
                        prob_naik = float(model.predict_proba(X_input)[0][1])
                        
                        if turnover < min_turnover:
                            aksi_final = "⚠️ LOW LIQUIDITY"
                        elif prob_naik >= conf_threshold:
                            aksi_final = "🔥 STRONG BUY"
                        else:
                            aksi_final = "⏳ HOLD / WAIT"
                            
                        harga_sekarang = latest['Close']
                        atr_sekarang = latest['ATR']
                        
                        tp = harga_sekarang + (metadata['tp_atr_mult'] * atr_sekarang)
                        sl = harga_sekarang - (metadata['sl_atr_mult'] * atr_sekarang)
                        
                        b_ratio = metadata['tp_atr_mult'] / metadata['sl_atr_mult']
                        f_kelly = (prob_naik * b_ratio - (1.0 - prob_naik)) / b_ratio
                        alokasi_kelly = max(0.0, min(f_kelly * 0.5, 0.15)) * 100 if aksi_final == "🔥 STRONG BUY" else 0.0
                        
                        all_results.append({
                            "KODE": t.replace(".JK", ""),
                            "CONFIDENCE": prob_naik * 100,
                            "AKSI": aksi_final,
                            "HARGA": int(harga_sekarang),
                            "TARGET PROFIT": int(tp),
                            "STOP LOSS": int(sl),
                            "ALOKASI KELLY": f"{alokasi_kelly:.1f}%",
                            "MAX HOLD": "10 Hari Kerja",
                            "TARGET EXIT": target_exit_str,
                            "TURNOVER": turnover
                        })
                    except Exception:
                        continue
            except Exception:
                continue
                
        status_text.empty()
        progress_bar.empty()
        
        st.session_state['scan_results'] = all_results

    if 'scan_results' in st.session_state and st.session_state['scan_results']:
        df_hasil = pd.DataFrame(st.session_state['scan_results'])
        df_buy = df_hasil[df_hasil['AKSI'] == "🔥 STRONG BUY"].sort_values(by="CONFIDENCE", ascending=False)
        df_hold = df_hasil[df_hasil['AKSI'] != "🔥 STRONG BUY"].sort_values(by="CONFIDENCE", ascending=False)
        
        st.markdown(f"### 🎯 HASIL PEMINDAIAN PASAR (TOTAL {len(df_hasil)} EMITEN)")
        
        if df_buy.empty:
            st.warning("🚨 Tidak ada saham yang memenuhi ambang batas sinyal Beli hari ini. Disarankan menyimpan cash.")
        else:
            st.markdown("#### 🏆 TOP RADAR MATCHES (KEYAKINAN PROBABILITAS TERTINGGI)")
            top_cards = min(3, len(df_buy))
            cols = st.columns(top_cards)
            
            for i in range(top_cards):
                row = df_buy.iloc[i]
                with cols[i]:
                    st.markdown(f"""
                    <div class="recom-card">
                        <span style="background-color: #10b981; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;">RANK #{i+1}</span>
                        <h3 style="margin: 8px 0 2px 0; color: #0f172a;">{row['KODE']}</h3>
                        <p style="margin: 0; font-size: 12px; color: #64748b;">Keyakinan Sinyal: <b style="color:#10b981; font-size:15px;">{row['CONFIDENCE']:.1f}%</b></p>
                        <hr style="margin: 10px 0; border: 0; border-top: 1px solid #e2e8f0;">
                        <table style="width:100%; font-size:12px; color:#334155;">
                            <tr><td><b>Harga Entry:</b></td><td style="text-align:right;">Rp {row['HARGA']}</td></tr>
                            <tr><td><b>Target Profit:</b></td><td style="text-align:right; color:#10b981;"><b>Rp {row['TARGET PROFIT']}</b></td></tr>
                            <tr><td><b>Stop Loss:</b></td><td style="text-align:right; color:#ef4444;"><b>Rp {row['STOP LOSS']}</b></td></tr>
                            <tr><td><b>Porsi Modal:</b></td><td style="text-align:right; color:#3b82f6;"><b>{row['ALOKASI KELLY']}</b></td></tr>
                            <tr><td><b>Est. Holding:</b></td><td style="text-align:right;"><b>10 Hari Kerja</b></td></tr>
                            <tr><td><b>Batas Waktu Exit:</b></td><td style="text-align:right; color:#d97706;"><b>{row['TARGET EXIT']}</b></td></tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📋 DAFTAR REKOMENDASI SAHAM STRONG BUY")
            st.dataframe(
                df_buy[["KODE", "CONFIDENCE", "HARGA", "TARGET PROFIT", "STOP LOSS", "ALOKASI KELLY", "MAX HOLD", "TARGET EXIT"]],
                use_container_width=True, hide_index=True
            )

        st.markdown("---")
        st.markdown("#### ⏳ DAFTAR EMITEN STATUS WAIT / HOLD / LOW LIQUIDITY")
        st.dataframe(
            df_hold[["KODE", "CONFIDENCE", "HARGA", "AKSI"]],
            use_container_width=True, hide_index=True
        )

# ------------------------------------------
# TAB 2: PORTFOLIO LIFECYCLE TRACKER
# ------------------------------------------
with tab_portfolio:
    st.markdown("### MANAJEMEN PORTOFOLIO AKTIF & TRACKER SIKLUS TRADING")
    st.info("**Aturan Eksekusi:** Begitu saham dibeli, abaikan skrining harian baru untuk emiten tersebut. Saham hanya dijual jika menyentuh **Target Profit**, **Stop Loss**, atau **Batas Waktu 10 Hari Kerja**.")
    
    portfolio = load_portfolio()
    
    # Form Input Transaksi Baru
    with st.expander("➕ **TAMBAHKAN TRANSAKSI POSISI BARU**", expanded=False):
        with st.form("form_add_position"):
            c1, c2, c3 = st.columns(3)
            with c1:
                in_kode = st.text_input("Kode Saham (contoh: BBCA):").upper().strip()
                in_buy_price = st.number_input("Harga Beli (Entry Price):", min_value=50, value=500, step=5)
            with c2:
                in_tp = st.number_input("Target Profit (TP):", min_value=50, value=560, step=5)
                in_sl = st.number_input("Stop Loss (SL):", min_value=50, value=460, step=5)
            with c3:
                in_date = st.date_input("Tanggal Beli:", value=date.today())
                in_lot = st.number_input("Jumlah Lot Beli:", min_value=1, value=10, step=1)
                
            submit_pos = st.form_submit_button("💾 Simpan Posisi ke Portofolio")
            
            if submit_pos:
                if not in_kode:
                    st.error("Kode saham wajib diisi!")
                else:
                    new_entry = {
                        "id": str(int(datetime.now().timestamp())),
                        "kode": in_kode,
                        "buy_price": float(in_buy_price),
                        "tp": float(in_tp),
                        "sl": float(in_sl),
                        "buy_date": in_date.strftime('%Y-%m-%d'),
                        "target_exit": calculate_target_date(in_date.strftime('%Y-%m-%d'), 10),
                        "lots": int(in_lot)
                    }
                    portfolio.append(new_entry)
                    save_portfolio(portfolio)
                    st.success(f"Posisi {in_kode} berhasil disimpan!")
                    st.rerun()

    # Pemonitoran Posisi Aktif
    if not portfolio:
        st.warning("Belum ada posisi aktif yang dicatat dalam portofolio.")
    else:
        st.markdown("#### 📊 DAFTAR POSISI BERJALAN & EVALUASI REAL-TIME")
        
        # Download data harga real-time emiten portofolio
        portfolio_tickers = [item['kode'] + ".JK" for item in portfolio]
        live_prices = {}
        
        try:
            p_data = yf.download(portfolio_tickers, period="5d", progress=False, group_by='ticker')
            for item in portfolio:
                t_symbol = item['kode'] + ".JK"
                try:
                    df_p = p_data[t_symbol].dropna(how='all') if len(portfolio_tickers) > 1 else p_data.dropna(how='all')
                    live_prices[item['kode']] = float(df_p['Close'].iloc[-1])
                except Exception:
                    live_prices[item['kode']] = item['buy_price']
        except Exception:
            for item in portfolio:
                live_prices[item['kode']] = item['buy_price']

        tracked_rows = []
        for idx, item in enumerate(portfolio):
            curr_price = live_prices.get(item['kode'], item['buy_price'])
            pnl_pct = ((curr_price - item['buy_price']) / item['buy_price']) * 100
            days_held = count_business_days_elapsed(item['buy_date'])
            
            # Evaluasi Aturan Exit
            if curr_price >= item['tp']:
                status_eval = "🎯 TAKE PROFIT HIT"
                badge_color = "background-color:#10b981; color:white;"
            elif curr_price <= item['sl']:
                status_eval = "🛑 STOP LOSS HIT"
                badge_color = "background-color:#ef4444; color:white;"
            elif days_held >= 10:
                status_eval = "⏰ TIME EXPIRED (CLOSE)"
                badge_color = "background-color:#f59e0b; color:white;"
            else:
                status_eval = "🔵 ACTIVE HOLD"
                badge_color = "background-color:#3b82f6; color:white;"

            tracked_rows.append({
                "ID": item['id'],
                "KODE": item['kode'],
                "TGL BELI": item['buy_date'],
                "ENTRY": f"Rp {item['buy_price']:,.0f}",
                "HARGA KINI": f"Rp {curr_price:,.0f}",
                "P&L (%)": f"{pnl_pct:+.2f}%",
                "TARGET PROFIT": f"Rp {item['tp']:,.0f}",
                "STOP LOSS": f"Rp {item['sl']:,.0f}",
                "HARI KERJA": f"{days_held} / 10 Hari",
                "BATAS EXIT": item['target_exit'],
                "REKOMENDASI EKSEKUSI": status_eval
            })

        df_portfolio = pd.DataFrame(tracked_rows)
        st.dataframe(
            df_portfolio.drop(columns=["ID"]),
            use_container_width=True, hide_index=True
        )

        st.markdown("---")
        st.markdown("#### 🗑️ HAPUS / TUTUP POSISI PORTOFOLIO")
        col_del1, col_del2 = st.columns([3, 1])
        with col_del1:
            option_to_delete = st.selectbox(
                "Pilih transaksi yang ingin dihapus/ditutup:",
                options=portfolio,
                format_func=lambda x: f"{x['kode']} | Beli: {x['buy_date']} | Rp {x['buy_price']}"
            )
        with col_del2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("❌ Hapus Posisi Ini", use_container_width=True):
                portfolio = [p for p in portfolio if p['id'] != option_to_delete['id']]
                save_portfolio(portfolio)
                st.success("Posisi berhasil dihapus dari portofolio!")
                st.rerun()