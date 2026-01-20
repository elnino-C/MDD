import streamlit as st
from pykrx import stock
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# --- [안전장치 함수] 1차원 시리즈로 강제 변환 ---
def force_series(data):
    """데이터가 DataFrame이거나 MultiIndex일 경우 강제로 1차원 Series로 변환"""
    if isinstance(data, pd.DataFrame):
        # 컬럼이 여러 개면 첫 번째 것만 가져옴
        return data.iloc[:, 0].squeeze()
    return data.squeeze()

# --- [함수] 지표 계산 ---
def calculate_rsi(prices, period=14):
    prices = force_series(prices) # 안전장치 적용
    if len(prices) < period: return pd.Series([np.nan]*len(prices), index=prices.index)
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_mdd(prices):
    prices = force_series(prices) # 안전장치 적용
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    mdd = drawdown.min()
    return mdd, drawdown

def calculate_cagr(start_price, end_price, days):
    if days <= 0 or start_price == 0: return 0
    years = days / 365.0
    try:
        cagr = (end_price / start_price) ** (1 / years) - 1
    except:
        cagr = 0
    return cagr

# --- [메인] 앱 구성 ---
def main():
    st.set_page_config(page_title="주식 퀀트 분석기", layout="wide")
    st.title("📈 주식 MDD & 벤치마크 분석 (Fix Ver.)")
    st.markdown("---")

    # 1. 사이드바
    with st.sidebar:
        st.header("🔍 분석 조건")
        if st.button("🗑️ 차트 새로고침 (캐시 삭제)"):
            st.cache_data.clear()
            st.rerun()

        ticker = st.text_input("종목 코드", value="005930")
        today = datetime.date.today()
        start_date = st.date_input("시작일", value=today - datetime.timedelta(days=365*3))
        end_date = st.date_input("종료일", value=today)
        
        st.markdown("---")
        use_benchmark = st.checkbox("벤치마크 비교", value=True)
        benchmark_option = st.radio("지수 선택", ("KOSPI", "KOSDAQ"), index=0) if use_benchmark else None
        use_rsi = st.checkbox("RSI 표시", value=True)
        
        st.markdown("")
        run_btn = st.button("분석 실행 🚀", type="primary")

    if run_btn:
        start_str_pykrx = start_date.strftime("%Y%m%d")
        end_str_pykrx = end_date.strftime("%Y%m%d")
        start_str_yf = start_date.strftime("%Y-%m-%d")
        end_str_yf = end_date.strftime("%Y-%m-%d")

        with st.spinner("데이터 분석 중..."):
            try:
                # 1. 종목 데이터 (pykrx)
                name = stock.get_market_ticker_name(ticker)
                if not name: return st.error("종목 코드 확인 필요")

                df = stock.get_market_ohlcv_by_date(start_str_pykrx, end_str_pykrx, ticker)
                if df.empty: return st.error("데이터 없음")
                
                # [Fix] pykrx 결과가 중복 컬럼을 가질 경우 방지
                df = df.loc[:, ~df.columns.duplicated()] 
                df.index = pd.to_datetime(df.index)
                
                # '종가' 컬럼만 안전하게 추출
                close_series = force_series(df['종가'])
                data = pd.DataFrame({'Stock': close_series})

                # 2. 벤치마크 데이터 (yfinance)
                if use_benchmark:
                    try:
                        yf_ticker = "^KS11" if benchmark_option == "KOSPI" else "^KQ11"
                        df_index = yf.download(yf_ticker, start=start_str_yf, end=end_str_yf, progress=False)
                        
                        if not df_index.empty:
                            # [Fix] yfinance 최신 버전 MultiIndex 처리
                            if isinstance(df_index.columns, pd.MultiIndex):
                                try:
                                    # 종가만 추출 시도 ('Close' 또는 'Adj Close')
                                    bench_series = df_index['Close']
                                    if isinstance(bench_series, pd.DataFrame):
                                        bench_series = bench_series.iloc[:, 0]
                                except:
                                    bench_series = df_index.iloc[:, 0] # 실패하면 무조건 첫번째 컬럼
                            else:
                                bench_series = df_index['Close']
                            
                            # 1차원 데이터로 강제 변환
                            bench_series = force_series(bench_series)
                            
                            # Timezone 제거 및 병합
                            bench_series.index = pd.to_datetime(bench_series.index).tz_localize(None)
                            data['Benchmark'] = bench_series
                    except Exception as e:
                        st.warning(f"벤치마크 데이터 로드 실패: {e}")

                data = data.dropna()
                if len(data) < 2: return st.error("데이터 부족")

                # --- 지표 계산 (안전장치 적용된 Series 사용) ---
                stock_series = force_series(data['Stock'])
                
                # Rebase
                norm_stock = (stock_series / stock_series.iloc[0]) * 100
                mdd_stock, dd_stock = calculate_mdd(stock_series)
                days = (data.index[-1] - data.index[0]).days
                cagr_stock = calculate_cagr(stock_series.iloc[0], stock_series.iloc[-1], days)
                total_ret_stock = (stock_series.iloc[-1] / stock_series.iloc[0]) - 1

                norm_bench, mdd_bench, dd_bench = None, None, None
                
                if use_benchmark and 'Benchmark' in data.columns:
                    bench_series = force_series(data['Benchmark'])
                    
                    norm_bench = (bench_series / bench_series.iloc[0]) * 100
                    mdd_bench, dd_bench = calculate_mdd(bench_series)
                    cagr_bench = calculate_cagr(bench_series.iloc[0], bench_series.iloc[-1], days)
                    total_ret_bench = (bench_series.iloc[-1] / bench_series.iloc[0]) - 1

                rsi = calculate_rsi(stock_series) if use_rsi else None
                current_rsi = rsi.iloc[-1] if use_rsi and not rsi.isna().all() else 50

                # --- 결과 출력 ---
                st.subheader(f"📊 {name} 분석 결과")
                c1, c2, c3, c4 = st.columns(4)
                
                ret_delta = f"{(total_ret_stock-total_ret_bench)*100:.1f}%p" if norm_bench is not None else None
                c1.metric("수익률", f"{total_ret_stock*100:.1f}%", ret_delta)
                
                cagr_delta = f"{(cagr_stock-cagr_bench)*100:.1f}%p" if norm_bench is not None else None
                c2.metric("CAGR", f"{cagr_stock*100:.1f}%", cagr_delta)
                
                mdd_delta = f"지수 {mdd_bench*100:.1f}%" if norm_bench is not None else None
                c3.metric("MDD", f"{mdd_stock*100:.1f}%", mdd_delta, delta_color="inverse")
                
                if use_rsi:
                    rsi_state = "과매수" if current_rsi > 70 else "과매도" if current_rsi < 30 else "중립"
                    c4.metric("RSI", f"{current_rsi:.1f}", rsi_state)

                # --- 차트 그리기 ---
                rows = 3 if use_rsi else 2
                row_heights = [0.5, 0.25, 0.25] if use_rsi else [0.6, 0.4]
                
                fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)

                # 1. 수익률
                fig.add_trace(go.Scatter(x=data.index, y=norm_stock, name=name, line=dict(color='red', width=1.5), hovertemplate=None), row=1, col=1)
                if norm_bench is not None:
                    fig.add_trace(go.Scatter(x=data.index, y=norm_bench, name=benchmark_option, line=dict(color='gray', dash='dash'), hovertemplate=None), row=1, col=1)

                # 2. Drawdown
                fig.add_trace(go.Scatter(x=dd_stock.index, y=dd_stock, name=f'{name} DD', line=dict(color='blue', width=1), fill='tozeroy', hovertemplate=None), row=2, col=1)
                if dd_bench is not None:
                    fig.add_trace(go.Scatter(x=dd_bench.index, y=dd_bench, name=f'{benchmark_option} DD', line=dict(color='gray', width=1, dash='dot'), hovertemplate=None), row=2, col=1)

                # 3. RSI
                if use_rsi:
                    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name='RSI', line=dict(color='purple'), hovertemplate=None), row=3, col=1)
                    fig.add_shape(type="line", x0=rsi.index[0], x1=rsi.index[-1], y0=70, y1=70, line=dict(color="red", dash="dash"), row=3, col=1)
                    fig.add_shape(type="line", x0=rsi.index[0], x1=rsi.index[-1], y0=30, y1=30, line=dict(color="blue", dash="dash"), row=3, col=1)

                # 호버 모드 및 레이아웃 설정
                fig.update_layout(
                    height=800 if use_rsi else 600,
                    hovermode="x unified",
                    showlegend=True,
                    margin=dict(t=50, b=20, l=10, r=10),
                    legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
                )
                fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='solid', spikecolor='grey')
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(tickformat=".0%", row=2, col=1)
                if use_rsi: fig.update_yaxes(range=[0, 100], row=3, col=1)

                st.plotly_chart(fig, use_container_width=True, theme=None)

            except Exception as e:
                st.error("분석 중 오류가 발생했습니다.")
                st.code(f"Error details: {e}") # 에러 메시지를 코드로 자세히 보여줌

if __name__ == "__main__":
    main()
