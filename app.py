import streamlit as st
from pykrx import stock
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime


# --- [함수] 지표 계산 ---
def calculate_rsi(prices, period=14):
    if len(prices) < period: return pd.Series([np.nan] * len(prices), index=prices.index)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_mdd(prices):
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

    # [변경 확인용] 제목에 v2.0 표시
    st.title("📈 주식 MDD & 벤치마크 분석 (v2.0)")
    st.markdown("---")

    # 1. 사이드바
    with st.sidebar:
        st.header("🔍 분석 조건")

        # [NEW] 캐시 삭제 버튼 (변화가 없을 때 사용)
        if st.button("🗑️ 캐시 데이터 지우기"):
            st.cache_data.clear()
            st.rerun()

        ticker = st.text_input("종목 코드", value="005930", help="예: 삼성전자(005930)")

        today = datetime.date.today()
        start_date = st.date_input("시작일", value=today - datetime.timedelta(days=365 * 3))
        end_date = st.date_input("종료일", value=today)

        st.markdown("---")
        st.subheader("⚙️ 옵션 설정")

        use_benchmark = st.checkbox("벤치마크(지수) 비교", value=True)
        if use_benchmark:
            benchmark_option = st.radio("비교할 지수", ("KOSPI", "KOSDAQ"), index=0)
        else:
            benchmark_option = None

        use_mdd_detail = st.checkbox("MDD 주요 지점 표시", value=True)
        use_rsi = st.checkbox("RSI(보조지표) 표시", value=True)

        st.markdown("")
        run_btn = st.button("분석 실행 🚀", type="primary")

    if run_btn:
        start_str_pykrx = start_date.strftime("%Y%m%d")
        end_str_pykrx = end_date.strftime("%Y%m%d")
        start_str_yf = start_date.strftime("%Y-%m-%d")
        end_str_yf = end_date.strftime("%Y-%m-%d")

        with st.spinner(f"'{ticker}' 데이터를 분석 중입니다..."):
            try:
                # 1. 종목 데이터 (pykrx)
                name = stock.get_market_ticker_name(ticker)
                if not name:
                    st.error(f"존재하지 않는 종목 코드입니다: {ticker}")
                    return

                df = stock.get_market_ohlcv_by_date(start_str_pykrx, end_str_pykrx, ticker)
                if df.empty:
                    st.error("❗ 해당 기간에 종목 데이터가 없습니다.")
                    return

                df.index = pd.to_datetime(df.index)
                data = pd.DataFrame({'Stock': df['종가']})

                # 2. 벤치마크 데이터 (yfinance)
                if use_benchmark:
                    try:
                        yf_ticker = "^KS11" if benchmark_option == "KOSPI" else "^KQ11"
                        df_index = yf.download(yf_ticker, start=start_str_yf, end=end_str_yf, progress=False)

                        if not df_index.empty:
                            if isinstance(df_index.columns, pd.MultiIndex):
                                benchmark_series = df_index['Close'][yf_ticker]
                            else:
                                benchmark_series = df_index['Close']

                            benchmark_series.index = pd.to_datetime(benchmark_series.index).tz_localize(None)
                            data['Benchmark'] = benchmark_series
                        else:
                            st.warning("벤치마크 데이터를 가져오지 못했습니다.")
                            use_benchmark = False
                    except Exception:
                        st.warning("지수 데이터 로드 실패. 비교를 생략합니다.")
                        use_benchmark = False

                data = data.dropna()
                if data.empty or len(data) < 2:
                    st.error("❗ 분석할 데이터가 부족합니다.")
                    return

                # 3. 지표 계산
                # (1) 기본 데이터 (Rebased)
                norm_stock = (data['Stock'] / data['Stock'].iloc[0]) * 100
                mdd_stock, dd_stock = calculate_mdd(data['Stock'])

                days = (data.index[-1] - data.index[0]).days
                cagr_stock = calculate_cagr(data['Stock'].iloc[0], data['Stock'].iloc[-1], days)
                total_ret_stock = (data['Stock'].iloc[-1] / data['Stock'].iloc[0]) - 1

                # (2) MDD 상세
                pre_mdd_peak_date, mdd_date, recovery_date = None, None, None
                if use_mdd_detail:
                    peak = data['Stock'].cummax()
                    mdd_date = dd_stock.idxmin()
                    peak_price_at_mdd = peak.loc[mdd_date]
                    pre_mdd_peak_date = peak[peak == peak_price_at_mdd].index[0]
                    post_mdd = data['Stock'].loc[mdd_date:]
                    recovery_candidates = post_mdd[post_mdd >= peak_price_at_mdd].index
                    recovery_candidates = recovery_candidates[recovery_candidates > mdd_date]
                    recovery_date = recovery_candidates[0] if not recovery_candidates.empty else None

                # (3) 벤치마크
                if use_benchmark:
                    norm_bench = (data['Benchmark'] / data['Benchmark'].iloc[0]) * 100
                    mdd_bench, dd_bench = calculate_mdd(data['Benchmark'])
                    cagr_bench = calculate_cagr(data['Benchmark'].iloc[0], data['Benchmark'].iloc[-1], days)
                    total_ret_bench = (data['Benchmark'].iloc[-1] / data['Benchmark'].iloc[0]) - 1

                # (4) RSI
                if use_rsi:
                    rsi = calculate_rsi(data['Stock'])
                    current_rsi = rsi.iloc[-1] if not rsi.isna().all() else 50

                # 4. 결과 출력
                st.subheader(f"📊 {name} 성과 분석")
                col1, col2, col3, col4 = st.columns(4)

                delta_ret = f"{(total_ret_stock - total_ret_bench) * 100:.1f}%p" if use_benchmark else None
                col1.metric("총 수익률", f"{total_ret_stock * 100:.1f}%", delta_ret)

                delta_cagr = f"{(cagr_stock - cagr_bench) * 100:.1f}%p" if use_benchmark else None
                col2.metric("CAGR (연평균)", f"{cagr_stock * 100:.1f}%", delta_cagr)

                delta_mdd = f"지수 MDD {mdd_bench * 100:.1f}%" if use_benchmark else None
                col3.metric("MDD (최대낙폭)", f"{mdd_stock * 100:.1f}%", delta_mdd, delta_color="inverse")

                if use_rsi:
                    rsi_state = "과매수" if current_rsi > 70 else "과매도" if current_rsi < 30 else "중립"
                    col4.metric("RSI (14일)", f"{current_rsi:.1f}", rsi_state)
                else:
                    col4.empty()

                # 5. 차트 그리기 (Plotly)
                rows = 3 if use_rsi else 2
                row_heights = [0.5, 0.25, 0.25] if use_rsi else [0.6, 0.4]

                fig = make_subplots(
                    rows=rows, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=row_heights,
                    subplot_titles=("수익률", "Drawdown", "RSI" if use_rsi else None)
                )

                # [Chart 1] 수익률
                # 중요: hovertemplate=None 으로 명시적 설정
                fig.add_trace(go.Scatter(
                    x=data.index, y=norm_stock, mode='lines', name=f"{name} (Rebased)",
                    line=dict(color='red', width=1.5), hovertemplate=None
                ), row=1, col=1)

                if use_benchmark:
                    fig.add_trace(go.Scatter(
                        x=data.index, y=norm_bench, mode='lines', name=f"{benchmark_option} (Rebased)",
                        line=dict(color='gray', dash='dash'), hovertemplate=None
                    ), row=1, col=1)

                # MDD 마커
                if use_mdd_detail and mdd_date is not None:
                    fig.add_trace(go.Scatter(
                        x=[pre_mdd_peak_date], y=[norm_stock.loc[pre_mdd_peak_date]],
                        mode='markers', name='하락 시작점', marker=dict(symbol='triangle-up', size=12, color='red'),
                        hoverinfo='skip'
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=[mdd_date], y=[norm_stock.loc[mdd_date]],
                        mode='markers', name='MDD 바닥', marker=dict(symbol='circle', size=10, color='black'),
                        hoverinfo='skip'
                    ), row=1, col=1)
                    if recovery_date:
                        fig.add_trace(go.Scatter(
                            x=[recovery_date], y=[norm_stock.loc[recovery_date]],
                            mode='markers', name='회복 완료점', marker=dict(symbol='triangle-down', size=12, color='green'),
                            hoverinfo='skip'
                        ), row=1, col=1)

                # [Chart 2] Drawdown
                fig.add_trace(go.Scatter(
                    x=dd_stock.index, y=dd_stock, mode='lines', name=f'{name} Drawdown',
                    line=dict(color='blue', width=1), fill='tozeroy', hovertemplate=None
                ), row=2, col=1)

                if use_benchmark:
                    fig.add_trace(go.Scatter(
                        x=dd_bench.index, y=dd_bench, mode='lines', name=f'{benchmark_option} Drawdown',
                        line=dict(color='gray', width=1, dash='dot'), hovertemplate=None
                    ), row=2, col=1)

                # [Chart 3] RSI
                if use_rsi:
                    fig.add_trace(go.Scatter(
                        x=rsi.index, y=rsi, mode='lines', name='RSI (14)',
                        line=dict(color='purple'), hovertemplate=None
                    ), row=3, col=1)
                    fig.add_shape(type="line", x0=rsi.index[0], x1=rsi.index[-1], y0=70, y1=70,
                                  line=dict(color="red", dash="dash", width=1), row=3, col=1)
                    fig.add_shape(type="line", x0=rsi.index[0], x1=rsi.index[-1], y0=30, y1=30,
                                  line=dict(color="blue", dash="dash", width=1), row=3, col=1)

                # --- [핵심] 레이아웃 설정 (Unified Hover 적용) ---
                fig.update_layout(
                    height=800 if use_rsi else 600,
                    hovermode="x unified",  # 핵심: x축 기준 통합 호버
                    showlegend=True,
                    margin=dict(l=20, r=20, t=60, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                # 크로스헤어(Spike Line) 설정
                fig.update_xaxes(
                    showspikes=True,
                    spikemode='across',
                    spikesnap='cursor',
                    spikedash='solid',
                    spikecolor='grey',
                    spikethickness=1
                )

                # Y축 포맷 설정
                fig.update_yaxes(title_text="Price (100 base)", row=1, col=1)
                fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)
                if use_rsi:
                    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

                # [중요] theme=None을 추가하여 Streamlit 스타일 간섭 방지
                st.plotly_chart(fig, use_container_width=True, theme=None)

            except Exception as e:
                st.error("오류가 발생했습니다.")
                st.warning(f"에러 상세: {e}")


if __name__ == "__main__":
    main()