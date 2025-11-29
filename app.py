import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
from groq import Groq

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False

st.set_page_config(page_title="African Economic Intelligence", layout="wide", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); 
        padding: 20px; border-radius: 10px; color: white; text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
    }
    .indicator-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); 
        padding: 15px; border-radius: 8px; color: white; margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .recovery-card {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%); 
        padding: 20px; border-radius: 10px; color: white; text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    .recovery-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
    }
    .health-score {font-size: 2.5rem; font-weight: bold; color: white;}
    .health-label {font-size: 0.9rem; margin-top: 8px; opacity: 0.95; color: white;}
    .status-good {background: linear-gradient(135deg, #065f46 0%, #047857 100%);}
    .status-fair {background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);}
    .status-poor {background: linear-gradient(135deg, #b91c1c 0%, #dc2626 100%);}
    .alert-box {
        background: #422006; 
        border-left: 4px solid #f59e0b; 
        padding: 15px; 
        border-radius: 5px; 
        margin: 10px 0;
        color: #fbbf24;
    }
    .section-header {
        border-bottom: 3px solid #1e40af; 
        padding-bottom: 10px; 
        margin: 30px 0 20px 0;
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
    }
    .info-card {
        background: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
        color: #e2e8f0;
        font-weight: 500;
    }
    .info-card strong {
        color: #ffffff;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #1e293b;
        border-radius: 5px 5px 0px 0px;
        font-weight: 500;
        color: #e2e8f0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e40af;
        color: white;
    }
    div[data-testid="stExpander"] {
        background-color: #1e293b;
        border-radius: 8px;
        border: 1px solid #334155;
    }
    div[data-testid="stExpander"] summary {
        color: #ffffff;
        font-weight: 600;
    }
    .help-text {
        font-size: 0.85rem;
        color: #94a3b8;
        font-style: italic;
        margin-top: 5px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    p, span, label {
        color: #e2e8f0;
    }
    .stMarkdown {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
for key in ['chat_history', 'df', 'anomalies', 'reports', 'data_loaded']:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ['chat_history', 'anomalies', 'reports'] else None
    if key == 'data_loaded':
        st.session_state[key] = False

@st.cache_data
def load_data(filepath):
    """Load economic data from CSV"""
    try:
        df = pd.read_csv(filepath)
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        df = df.dropna(subset=['Time', 'Country', 'Indicator', 'Amount'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_health_score(country_data, selected_date):
    country_data = country_data[country_data['Time'] <= selected_date].copy()
    if len(country_data) == 0:
        return None, {}

    weights = {
        'GDP Growth Rate': 0.25, 
        'Inflation Rate': 0.20, 
        'Unemployment Rate': 0.20,
        'Budget Deficit/Surplus': 0.15, 
        'Government Debt': 0.10, 
        'Exports': 0.10
    }

    components = {}
    total_weight = 0
    weighted_sum = 0

    # helper for safe minmax scoring that never returns 0
    def scaled_score(value, low, high):
        # clamp into range
        v = max(min(value, high), low)
        # scale to 0..1
        scaled = (v - low) / (high - low)
        # convert to 20..100
        return 20 + scaled * 80

    for indicator, weight in weights.items():
        ind_data = country_data[country_data['Indicator'] == indicator]
        if len(ind_data) == 0:
            continue

        latest_value = ind_data.sort_values('Time').iloc[-1]['Amount']

        if indicator == 'GDP Growth Rate':
            score = scaled_score(latest_value, -5, 10)
        elif indicator == 'Inflation Rate':
            score = scaled_score(latest_value, 0, 20)
        elif indicator == 'Unemployment Rate':
            score = scaled_score(latest_value, 0, 25)
        elif indicator == 'Budget Deficit/Surplus':
            score = scaled_score(latest_value, -15, 10)
        elif indicator == 'Government Debt':
            score = scaled_score(latest_value, 30, 200)
        elif indicator == 'Exports':
            score = scaled_score(latest_value, 0, 100)
        else:
            score = 60

        components[indicator] = {
            'value': latest_value,
            'score': score
        }

        weighted_sum += score * weight
        total_weight += weight

    if total_weight == 0:
        return None, components

    final_score = weighted_sum / total_weight

    return final_score, components


def calculate_recovery_index(country_data, base_date, current_date):
    country_data = country_data.copy()

    baseline_data = country_data[country_data['Time'] <= base_date]
    current_data = country_data[
        (country_data['Time'] > base_date) &
        (country_data['Time'] <= current_date)
    ]

    if len(baseline_data) == 0 or len(current_data) == 0:
        return None, {}

    indicators = ['GDP Growth Rate', 'Unemployment Rate', 'Inflation Rate', 'Exports']

    recovery_components = {}
    weighted_sum = 0
    total_weight = 0

    # simple equal weights
    weight = 1 / len(indicators)

    # helper: map "improvement ratio" into nice 20..100
    def scaled_recovery(ratio):
        ratio = max(min(ratio, 1), -1)  # clamp to -1..1
        return 20 + (ratio + 1) * 40    # -1 -> 20, 0 -> 60, +1 -> 100

    for indicator in indicators:
        base = baseline_data[baseline_data['Indicator'] == indicator]
        curr = current_data[current_data['Indicator'] == indicator]

        if len(base) == 0 or len(curr) == 0:
            continue

        base_val = base.sort_values('Time').iloc[-1]['Amount']
        curr_val = curr.sort_values('Time').iloc[-1]['Amount']

        if indicator == 'GDP Growth Rate':
            # positive change is good
            if base_val == 0:
                ratio = 0
            else:
                ratio = (curr_val - base_val) / (abs(base_val) + 1)

        elif indicator == 'Unemployment Rate':
            # decrease is good
            if base_val == 0:
                ratio = 0
            else:
                ratio = (base_val - curr_val) / (base_val + 1)

        elif indicator == 'Inflation Rate':
            # decrease is generally good
            if base_val == 0:
                ratio = 0
            else:
                ratio = (base_val - curr_val) / (base_val + 1)

        elif indicator == 'Exports':
            # increase is good
            if base_val == 0:
                ratio = 0
            else:
                ratio = (curr_val - base_val) / (abs(base_val) + 1)

        recovery_pct = scaled_recovery(ratio)

        recovery_components[indicator] = {
            'base_value': base_val,
            'current_value': curr_val,
            'change': curr_val - base_val,
            'ratio': ratio,
            'recovery_pct': recovery_pct
        }

        weighted_sum += recovery_pct * weight
        total_weight += weight

    if total_weight == 0:
        return None, recovery_components

    recovery_score = weighted_sum / total_weight
    return recovery_score, recovery_components


def get_health_status(score):
    """Convert score to readable status"""
    if score >= 70:
        return "Strong", "status-good"
    elif score >= 50:
        return "Moderate", "status-fair"
    else:
        return "Weak", "status-poor"

def detect_anomalies(df, threshold=2.5):
    """Detect statistical anomalies in economic indicators"""
    anomalies = []
    for country in df['Country'].dropna().unique():
        for indicator in df['Indicator'].dropna().unique():
            data = df[(df['Country'] == country) & (df['Indicator'] == indicator)].copy()
            if len(data) < 5:
                continue
            
            mean, std = data['Amount'].mean(), data['Amount'].std()
            if std == 0:
                continue
            
            data['Z_Score'] = (data['Amount'] - mean) / std
            anomaly_data = data[abs(data['Z_Score']) > threshold]
            
            for _, row in anomaly_data.iterrows():
                anomalies.append({
                    'Country': country, 
                    'Indicator': indicator, 
                    'Date': row['Time'],
                    'Value': row['Amount'], 
                    'Z_Score': row['Z_Score'],
                    'Severity': 'High' if abs(row['Z_Score']) > 3 else 'Medium'
                })
    return anomalies

def forecast_prophet(data, country, indicator, periods=12):
    """Generate forecast using Prophet if available"""
    if not PROPHET_AVAILABLE:
        return forecast_simple(data, country, indicator, periods)
    
    try:
        ind_data = data[(data['Country'] == country) & (data['Indicator'] == indicator)].copy()
        if len(ind_data) < 12:
            return forecast_simple(data, country, indicator, periods)
        
        df_prophet = ind_data.sort_values('Time')[['Time', 'Amount']].copy()
        df_prophet.columns = ['ds', 'y']
        
        model = Prophet(yearly_seasonality=False, daily_seasonality=False, interval_width=0.95)
        with st.spinner('Training forecast model...'):
            model.fit(df_prophet)
        
        future = model.make_future_dataframe(periods=periods, freq='MS')
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    except Exception as e:
        st.warning(f"Prophet forecast failed, using simple forecast: {str(e)}")
        return forecast_simple(data, country, indicator, periods)

def forecast_simple(data, country, indicator, periods=12):
    """Fallback simple moving average forecast"""
    ind_data = data[(data['Country'] == country) & (data['Indicator'] == indicator)].copy()
    if len(ind_data) < 10:
        return None
    
    ind_data = ind_data.sort_values('Time')
    values = ind_data['Amount'].tail(6).values
    forecast_value = np.mean(values)
    std_dev = np.std(values)
    
    last_date = ind_data['Time'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='MS')
    
    return pd.DataFrame({
        'ds': future_dates,
        'yhat': [forecast_value] * periods,
        'yhat_lower': [forecast_value - 1.96*std_dev] * periods,
        'yhat_upper': [forecast_value + 1.96*std_dev] * periods
    })

def generate_chart_insight(df_filtered, selected, indicator, chart_type, api_key):
    """Generate AI insight for chart data"""
    if not api_key:
        return None
    
    try:
        # Prepare data summary for the AI
        summary_data = []
        for country in selected:
            country_data = df_filtered[(df_filtered['Country'] == country) & (df_filtered['Indicator'] == indicator)]
            if len(country_data) > 0:
                latest = country_data.sort_values('Time').iloc[-1]
                earliest = country_data.sort_values('Time').iloc[0]
                avg = country_data['Amount'].mean()
                summary_data.append(f"{country}: Latest={latest['Amount']:.2f}, Earliest={earliest['Amount']:.2f}, Average={avg:.2f}")
        
        if not summary_data:
            return None
        
        context = "\n".join(summary_data)
        
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert economist. Provide brief, insightful analysis of economic data in 2-3 sentences."},
                {"role": "user", "content": f"Analyze this {indicator} data for {', '.join(selected)}:\n{context}\n\nProvide a brief insight about trends, comparisons, and what this means economically."}
            ],
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return None

def get_ai_insights(question, df, api_key, selected_countries):
    """Generate AI-powered insights using Groq"""
    if not api_key:
        return "Please configure your Groq API key to use AI features."
    
    try:
        client = Groq(api_key=api_key)
        
        # Filter data for selected countries only
        country_data = df[df['Country'].isin(selected_countries)]
        latest_data = country_data.sort_values('Time').groupby(['Country', 'Indicator']).last().reset_index()
        context = latest_data.head(50).to_string()
        
        countries_str = ", ".join(selected_countries)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"You are an expert economist analyzing African economic data. Focus your analysis on the following selected countries: {countries_str}. Dataset context:\n{context}"},
                {"role": "user", "content": question}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def create_correlation_heatmap(df, countries, indicator):
    pivot_data = df[(df['Country'].isin(countries)) & (df['Indicator'] == indicator)]\
        .pivot_table(index='Time', columns='Country', values='Amount')
    
    if pivot_data.empty or len(pivot_data) < 2:
        return None
    
    corr = pivot_data.corr()

    # Convert corr values to text for annotation
    text_labels = [[str(round(val, 2)) for val in row] for row in corr.values]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title="Correlation"),
        text=text_labels,
        texttemplate="%{text}",
        textfont={"size":12, "color":"black"},
    ))

    fig.update_layout(
        title=f'{indicator} - Correlation Matrix',
        height=500
    )
    return fig

def create_box_plot(df, countries, indicator):
    """Create distribution box plot"""
    plot_data = df[(df['Country'].isin(countries)) & (df['Indicator'] == indicator)]
    if len(plot_data) == 0:
        return None
    
    fig = go.Figure()
    for country in countries:
        country_data = plot_data[plot_data['Country'] == country]
        if len(country_data) > 0:
            fig.add_trace(go.Box(y=country_data['Amount'], name=country, boxmean='sd'))
    
    fig.update_layout(
        title=f'{indicator} - Distribution by Country', 
        yaxis_title='Value', 
        height=500,
        hovermode='y unified'
    )
    return fig

def predict_downturn(df, country):
    """Predict economic downturn risk"""
    gdp = df[(df['Country'] == country) & (df['Indicator'] == 'GDP Growth Rate')].sort_values('Time')
    if len(gdp) < 4:
        return {'risk_score': 0, 'risk_level': 'Insufficient Data', 'risk_factors': ['Less than 4 data points']}
    
    recent = gdp.tail(4)['Amount'].values
    slope = np.polyfit(range(len(recent)), recent, 1)[0]
    
    risk_score, risk_factors = 0, []
    
    if slope < -0.5:
        risk_score += 30
        risk_factors.append(f"Declining GDP trend (slope: {slope:.2f})")
    
    if recent[-1] < 2.0:
        risk_score += 40
        risk_factors.append(f"Low GDP growth: {recent[-1]:.2f}%")
    
    inflation = df[(df['Country'] == country) & (df['Indicator'] == 'Inflation Rate')].sort_values('Time')
    if len(inflation) > 0 and inflation.iloc[-1]['Amount'] > 10:
        risk_score += 30
        risk_factors.append(f"High inflation: {inflation.iloc[-1]['Amount']:.2f}%")
    
    risk_level = "Low" if risk_score < 30 else "Medium" if risk_score < 60 else "High"
    return {'risk_score': risk_score, 'risk_level': risk_level, 'risk_factors': risk_factors}

def export_csv(df):
    """Export dataframe to CSV"""
    return df.to_csv(index=False).encode('utf-8')

def generate_report_text(country, health_score, components, df):
    """Generate comprehensive text report"""
    status, _ = get_health_status(health_score)
    report = f"""ECONOMIC INTELLIGENCE REPORT
{'='*60}
Country: {country}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

ECONOMIC HEALTH ASSESSMENT
Economic Health Score: {health_score:.1f}/100
Status: {status}

KEY INDICATORS:
"""
    for indicator, data in components.items():
        report += f"\n  {indicator}: {data['value']:.2f} (Component Score: {data['score']:.1f})"
    
    risk = predict_downturn(df, country)
    report += f"\n\nRISK ASSESSMENT\nOverall Risk Level: {risk['risk_level']} ({risk['risk_score']}/100)"
    report += f"\nIdentified Risk Factors:\n"
    for factor in risk['risk_factors']:
        report += f"  - {factor}\n"
    
    return report

def main():
    # Main header
    st.markdown('<h1 style="color:#ffffff;">📊 African Economic Intelligence Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Real-time economic monitoring, AI-powered insights, and predictive analytics for African nations**")
    
    # API Key input at the top
    api_key = "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"
    
    st.divider()

    # Load data
    if st.session_state.df is None:
        with st.spinner("🔄 Loading economic data..."):
            df = load_data("data.csv")
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success(f"✅ Data loaded successfully: {len(df):,} records from {len(df['Country'].unique())} countries")
            else:
                st.error("❌ Failed to load data.csv. Please ensure the file exists in the same directory.")
                st.info("💡 **Tip:** Make sure your data.csv file has columns: Time, Country, Indicator, Amount")
                return
    else:
        df = st.session_state.df
    
    # Filters
    st.markdown("### 🎯 Filter Your Analysis")
    col1, col2, col3 = st.columns([2, 2, 1.5])
    with col1:
        countries = sorted([c for c in df['Country'].dropna().unique() if pd.notna(c)])
        
        # Default countries
        default_countries = ['Algeria', 'Nigeria']
        # Filter to only include countries that exist in the dataset
        default_selection = [c for c in default_countries if c in countries]
        
        # Add "Select All" checkbox
        select_all = st.checkbox("Select All Countries", value=False)
        
        if select_all:
            selected = st.multiselect(
                "📍 Select Countries",
                countries, 
                default=countries,
                help="Choose one or more countries to analyze"
            )
        else:
            selected = st.multiselect(
                "📍 Select Countries",
                countries, 
                default=default_selection,
                help="Choose one or more countries to analyze"
            )
    
    with col2:
        date_preset = st.selectbox(
            "📅 Time Range",
            ["All Time", "Last Year", "Last 5 Years", "Last Quarter"],
            help="Select the time period for your analysis"
        )
    
    with col3:
        max_date = df['Time'].max()
        if date_preset == "Last Year":
            filter_date = max_date - pd.DateOffset(years=1)
        elif date_preset == "Last 5 Years":
            filter_date = max_date - pd.DateOffset(years=5)
        elif date_preset == "Last Quarter":
            filter_date = max_date - pd.DateOffset(months=3)
        else:
            filter_date = df['Time'].min()
        
        selected_date = max_date
        st.metric("Data Points", f"{len(df[(df['Time'] >= filter_date) & (df['Country'].isin(selected))]):,}")
    
    df_filtered = df[(df['Time'] >= filter_date) & (df['Time'] <= selected_date) & (df['Country'].isin(selected))]
    
    st.divider()
    
    # Detect anomalies on load
    if not st.session_state.anomalies:
        st.session_state.anomalies = detect_anomalies(df_filtered)
    
    # Display anomaly alerts
    # if st.session_state.anomalies:
    #     high = [a for a in st.session_state.anomalies if a['Severity']=='High']
    #     if high:
    #         with st.expander(f"⚠️ {len(high)} High-Severity Anomalies Detected - Click to view", expanded=False):
    #             for anomaly in high[:5]:  # Show top 5
    #                 st.warning(f"**{anomaly['Country']}** - {anomaly['Indicator']}: {anomaly['Value']:.2f} (Z-Score: {anomaly['Z_Score']:.2f})")
    
    if selected:
        tabs = st.tabs([
            "📊 Overview", 
            "💬 AI Chat Assistant", 
            "📈 Trend Analysis", 
            "🔮 Forecasting", 
            "🚀 Recovery Tracker",
            "⚠️ Risk Monitor", 
            "📥 Export Data"
        ])
        
        with tabs[0]:  # Overview Dashboard
            st.markdown('<div class="section-header">Economic Health Overview</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-card">📌 <strong>Quick Summary:</strong> View the economic health scores and key indicators for your selected countries at a glance.</div>', unsafe_allow_html=True)
            
            for country in selected:
                c_data = df_filtered[df_filtered['Country'] == country]
                score, comp = calculate_health_score(c_data, selected_date)
                
                if score is not None:
                    with st.expander(f"🌍 {country} - Click to view details", expanded=(len(selected) <= 3)):
                        status, status_class = get_health_status(score)
                        
                        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                        
                        with col1:
                            st.markdown(f'''
                            <div class="metric-card {status_class}">
                                <div class="health-score">{score:.0f}</div>
                                <div class="health-label">Economic Health Score</div>
                                <div class="health-label">Status: {status}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        # Display top 3 available indicators
                        available_indicators = ['GDP Growth Rate', 'Inflation Rate', 'Unemployment Rate']
                        displayed_count = 0
                        for idx, ind in enumerate(available_indicators):
                            if displayed_count >= 3:
                                break
                            i_data = c_data[c_data['Indicator']==ind].sort_values('Time')
                            if len(i_data) > 0:
                                latest = i_data.iloc[-1]
                                prev = i_data.iloc[-2] if len(i_data) > 1 else None
                                change = latest['Amount'] - prev['Amount'] if prev is not None else 0
                                
                                with [col2, col3, col4][displayed_count]:
                                    st.metric(
                                        ind, 
                                        f"{latest['Amount']:.2f}",
                                        f"{change:+.2f}" if prev is not None else "N/A",
                                        help=f"Latest value as of {latest['Time'].strftime('%Y-%m-%d')}"
                                    )
                                displayed_count += 1
                        
                        # Generate AI insight for overview
                        if api_key:
                            with st.spinner("🤖 Generating AI insight..."):
                                try:
                                    client = Groq(api_key=api_key)
                                    comp_summary = ", ".join([f"{k}: {v['value']:.2f}" for k, v in comp.items()])
                                    response = client.chat.completions.create(
                                        model="llama-3.3-70b-versatile",
                                        messages=[
                                            {"role": "system", "content": "You are an expert economist. Provide brief, insightful analysis in 2-3 sentences."},
                                            {"role": "user", "content": f"Analyze {country}'s economic health (score: {score:.1f}/100). Key indicators: {comp_summary}. What are the main strengths and concerns?"}
                                        ],
                                        max_tokens=200
                                    )
                                    insight = response.choices[0].message.content
                                    st.markdown(f'<div class="info-card">🤖 <strong>AI Insight:</strong> {insight}</div>', unsafe_allow_html=True)
                                except:
                                    pass
                else:
                    st.info(f"ℹ️ Insufficient data available for {country} in the selected time period.")
        
        with tabs[1]:  # AI Chat Assistant
            st.markdown('<div class="section-header">AI-Powered Chat Assistant</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-card">🤖 <strong>Ask me anything!</strong> I can analyze data for: <strong>{", ".join(selected)}</strong><br>💡 Try asking: "What are the economic trends?", "Compare inflation rates", or "Which country is performing best?"</div>', unsafe_allow_html=True)
            
            if not api_key:
                st.warning("🔑 **Groq API Key Required**")
                st.markdown("""
                To enable AI-powered insights:
                1. Get a free API key from [Groq Console](https://console.groq.com)
                2. Enter it in the field at the top of the page
                3. Start chatting with the AI assistant!
                """)
            else:
                # Display chat history
                chat_container = st.container()
                with chat_container:
                    for msg in st.session_state.chat_history:
                        with st.chat_message("user"):
                            st.write(msg['q'])
                        with st.chat_message("assistant"):
                            st.write(msg['a'])
                
                # Input area
                st.divider()
                q = st.chat_input("Type your question here... (e.g., 'What are the key economic trends for Nigeria?')")
                
                if q:
                    # Calculate tokens (rough estimate: 1 token ≈ 4 characters)
                    current_tokens = sum(len(msg['q']) + len(msg['a']) for msg in st.session_state.chat_history) // 4
                    message_tokens = len(q) // 4
                    
                    if current_tokens + message_tokens > 9000:
                        st.error("⚠️ Token limit approaching. Please refresh the page to start a new conversation.")
                    else:
                        with st.spinner("🤔 Analyzing your question..."):
                            ans = get_ai_insights(q, df_filtered, api_key, selected)
                            st.session_state.chat_history.append({"q": q, "a": ans})
                            st.rerun()
                
                # Token counter with progress bar
                total_tokens = sum(len(msg['q']) + len(msg['a']) for msg in st.session_state.chat_history) // 4
                progress = min(total_tokens / 10000, 1.0)
                st.progress(progress)
                st.caption(f"💬 Conversation tokens: {total_tokens:,} / 10,000")
                
                if st.button("🔄 Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        with tabs[2]:  # Trend Analysis
            st.markdown('<div class="section-header">Economic Trend Analysis</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-card">📊 <strong>Visualize trends:</strong> Compare economic indicators across countries with interactive charts.</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                available_indicators = sorted([i for i in df_filtered['Indicator'].dropna().unique() if pd.notna(i)])
                # Set default to Consumer Price Index (CPI) if available
                default_indicator = 'Consumer Price Index (CPI)' if 'Consumer Price Index (CPI)' in available_indicators else available_indicators[0] if available_indicators else None
                
                ind = st.selectbox(
                    "📊 Select Economic Indicator", 
                    available_indicators,
                    index=available_indicators.index(default_indicator) if default_indicator in available_indicators else 0,
                    help="Choose an indicator to visualize trends"
                )
            with col2:
                chart_type = st.radio(
                    "📈 Visualization Type", 
                    ["Line Chart", "Correlation Map", "Distribution"], 
                    horizontal=False,
                    help="Select how you want to view the data"
                )
            
            st.info(f"ℹ️ Showing data for: **{', '.join(selected)}** | Indicator: **{ind}**")
            
            if chart_type == "Line Chart":
                fig = go.Figure()
                data_found = False
                for c in selected:
                    data = df_filtered[(df_filtered['Country']==c) & (df_filtered['Indicator']==ind)].sort_values('Time')
                    if len(data) > 0:
                        data_found = True
                        fig.add_trace(go.Scatter(
                            x=data['Time'], 
                            y=data['Amount'], 
                            mode='lines+markers', 
                            name=c,
                            line=dict(width=3),
                            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
                        ))
                
                if data_found:
                    fig.update_layout(
                        title=f'{ind} - Time Series Comparison', 
                        height=600, 
                        hovermode='x unified',
                        yaxis_title=ind,
                        xaxis_title='Date',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    st.plotly_chart(fig, width='stretch')
                    
                    # Generate AI insight
                    if api_key:
                        with st.spinner("🤖 Generating AI insight..."):
                            insight = generate_chart_insight(df_filtered, selected, ind, chart_type, api_key)
                            if insight:
                                st.markdown(f'<div class="info-card">🤖 <strong>AI Insight:</strong> {insight}</div>', unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ No data available for **{ind}** in the selected countries and time period.")
            
            elif chart_type == "Correlation Map":
                hm = create_correlation_heatmap(df_filtered, selected, ind)
                if hm:
                    st.plotly_chart(hm, width='stretch')
                    st.caption("💡 Values closer to 1 (red) indicate strong positive correlation, while values closer to -1 (blue) indicate negative correlation.")
                    
                    # Generate AI insight
                    if api_key:
                        with st.spinner("🤖 Generating AI insight..."):
                            insight = generate_chart_insight(df_filtered, selected, ind, chart_type, api_key)
                            if insight:
                                st.markdown(f'<div class="info-card">🤖 <strong>AI Insight:</strong> {insight}</div>', unsafe_allow_html=True)
                else:
                    st.info("ℹ️ Insufficient data for correlation analysis. Need at least 2 countries with overlapping data points.")
            
            else:  # Distribution
                bp = create_box_plot(df_filtered, selected, ind)
                if bp:
                    st.plotly_chart(bp, width='stretch')
                    st.caption("💡 The box shows the quartile ranges, while the whiskers show the data distribution. Outliers appear as individual points.")
                    
                    # Generate AI insight
                    if api_key:
                        with st.spinner("🤖 Generating AI insight..."):
                            insight = generate_chart_insight(df_filtered, selected, ind, chart_type, api_key)
                            if insight:
                                st.markdown(f'<div class="info-card">🤖 <strong>AI Insight:</strong> {insight}</div>', unsafe_allow_html=True)
                else:
                    st.info("ℹ️ Insufficient data for distribution analysis.")
        
        with tabs[3]:  # Forecasting
            st.markdown('<div class="section-header">Economic Forecasting Engine</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-card">🔮 <strong>Predict the future:</strong> Generate AI-powered forecasts for economic indicators using historical data patterns.</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                f_country = st.selectbox("🌍 Country", selected, key="forecast_country", help="Select country to forecast")
            with col2:
                available_indicators = sorted([i for i in df['Indicator'].dropna().unique() if pd.notna(i)])
                f_ind = st.selectbox("📊 Indicator", available_indicators, help="Choose indicator to forecast")
            with col3:
                periods = st.number_input("📅 Months Ahead", 3, 24, 12, help="Number of months to forecast")
            
            if st.button("🚀 Generate Forecast", width='stretch', type="primary"):
                with st.spinner("🔄 Training model and generating forecast..."):
                    forecast = forecast_prophet(df, f_country, f_ind, periods)
                    
                    if forecast is not None:
                        historical = df[(df['Country']==f_country) & (df['Indicator']==f_ind)].sort_values('Time')
                        
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=historical['Time'], 
                            y=historical['Amount'],
                            mode='lines+markers', 
                            name='Historical Data',
                            line=dict(color='#667eea', width=3),
                            hovertemplate='<b>Historical</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
                        ))
                        
                        # Forecast
                        forecast_sorted = forecast.sort_values('ds')
                        fig.add_trace(go.Scatter(
                            x=forecast_sorted['ds'], 
                            y=forecast_sorted['yhat'],
                            mode='lines+markers', 
                            name='Forecast',
                            line=dict(color='#f2994e', dash='dash', width=3),
                            hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
                        ))
                        
                        # Confidence interval
                        fig.add_trace(go.Scatter(
                            x=forecast_sorted['ds'],
                            y=forecast_sorted['yhat_upper'],
                            fill=None,
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            showlegend=False,
                            name='Upper Bound'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_sorted['ds'],
                            y=forecast_sorted['yhat_lower'],
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            name='95% Confidence Interval',
                            fillcolor='rgba(242, 153, 78, 0.2)'
                        ))
                        
                        fig.update_layout(
                            title=f"{f_country} - {f_ind} Forecast ({periods} months ahead)",
                            height=600,
                            hovermode='x unified',
                            yaxis_title=f_ind,
                            xaxis_title='Date',
                            xaxis=dict(rangeslider=dict(visible=False))
                        )
                        st.plotly_chart(fig, width='stretch')
                        
                        # Forecast summary
                        latest_forecast = forecast_sorted['yhat'].iloc[-1]
                        uncertainty = (forecast_sorted['yhat_upper'].iloc[-1] - forecast_sorted['yhat_lower'].iloc[-1]) / 2
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📊 Final Forecast Value", f"{latest_forecast:.2f}")
                        with col2:
                            st.metric("📉 Uncertainty Range", f"±{uncertainty:.2f}")
                        with col3:
                            confidence = "High" if uncertainty < abs(latest_forecast * 0.1) else "Medium" if uncertainty < abs(latest_forecast * 0.2) else "Low"
                            st.metric("✅ Confidence Level", confidence)
                        
                        # Generate AI insight for forecast
                        if api_key:
                            with st.spinner("🤖 Generating AI insight..."):
                                try:
                                    client = Groq(api_key=api_key)
                                    latest_hist = historical['Amount'].iloc[-1]
                                    response = client.chat.completions.create(
                                        model="llama-3.3-70b-versatile",
                                        messages=[
                                            {"role": "system", "content": "You are an expert economist. Provide brief forecast analysis in 2-3 sentences."},
                                            {"role": "user", "content": f"Analyze this {f_ind} forecast for {f_country}: Current value is {latest_hist:.2f}, forecast in {periods} months is {latest_forecast:.2f} (±{uncertainty:.2f}). What does this trend suggest?"}
                                        ],
                                        max_tokens=200
                                    )
                                    insight = response.choices[0].message.content
                                    st.markdown(f'<div class="info-card">🤖 <strong>AI Insight:</strong> {insight}</div>', unsafe_allow_html=True)
                                except:
                                    pass
                    else:
                        st.warning("⚠️ Insufficient historical data for forecasting. Need at least 10 data points.")
        
        with tabs[4]:  # Economic Recovery
            st.markdown('<div class="section-header">Economic Recovery Tracker</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-card">🚀 <strong>Track recovery progress:</strong> Measure economic recovery against baseline periods like COVID-19 or custom dates.</div>', unsafe_allow_html=True)
            
            # Recovery controls
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                recovery_country = st.selectbox("🌍 Country", selected, key="recovery_country", help="Select country to analyze")
            
            with col2:
                # Date range for recovery analysis
                min_date = df['Time'].min()
                recovery_end_date = st.date_input(
                    "📅 Analysis End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    help="End date for recovery analysis"
                )
                recovery_end_date = pd.to_datetime(recovery_end_date)
            
            with col3:
                # Base period selection
                base_period = st.selectbox(
                    "📍 Baseline Period",
                    ["COVID-19 (March 2020)", "Pre-Pandemic (Dec 2019)", "Custom Date"],
                    help="Choose the baseline to measure recovery against"
                )
                
                if base_period == "COVID-19 (March 2020)":
                    base_date = pd.to_datetime("2020-03-01")
                elif base_period == "Pre-Pandemic (Dec 2019)":
                    base_date = pd.to_datetime("2019-12-01")
                else:
                    custom_base = st.date_input(
                        "Select Base Date",
                        value=pd.to_datetime("2020-03-01"),
                        min_value=min_date,
                        max_value=max_date
                    )
                    base_date = pd.to_datetime(custom_base)
            
            # Calculate recovery index
            country_recovery_data = df[df['Country'] == recovery_country]
            recovery_index, recovery_components = calculate_recovery_index(
                country_recovery_data, 
                base_date, 
                recovery_end_date
            )
            
            if recovery_index is not None:
                st.divider()
                
                # Display recovery score
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    recovery_status = "Strong Recovery" if recovery_index >= 70 else "Moderate Recovery" if recovery_index >= 50 else "Weak Recovery"
                    recovery_color = "status-good" if recovery_index >= 70 else "status-fair" if recovery_index >= 50 else "status-poor"
                    
                    st.markdown(f'''
                    <div class="recovery-card {recovery_color}">
                        <div class="health-score">{recovery_index:.0f}</div>
                        <div class="health-label">Recovery Index</div>
                        <div class="health-label">{recovery_status}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Display indicator cards
                indicators_to_show = ['GDP Growth Rate', 'Unemployment Rate', 'Inflation Rate']
                card_cols = [col2, col3, col4]
                
                for idx, indicator in enumerate(indicators_to_show):
                    if indicator in recovery_components:
                        comp = recovery_components[indicator]
                        with card_cols[idx]:
                            st.markdown(f'''
                            <div class="indicator-card">
                                <div style="font-size: 1.5rem; font-weight: bold;">{comp['current_value']:.2f}</div>
                                <div style="font-size: 0.8rem; margin-top: 5px;">{indicator}</div>
                                <div style="font-size: 0.75rem; margin-top: 5px; opacity: 0.9;">
                                    Change: {comp['change']:+.2f}<br>
                                    Recovery: {comp['recovery_pct']:.0f}%
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                
                st.divider()
                
                # Recovery components details
                st.markdown("#### Recovery Components Details")
                recovery_df = pd.DataFrame([
                    {
                        'Indicator': ind,
                        'Baseline Value': comp['base_value'],
                        'Current Value': comp['current_value'],
                        'Change': comp['change'],
                        'Recovery %': comp['recovery_pct']
                    }
                    for ind, comp in recovery_components.items()
                ])
                st.dataframe(recovery_df, width='stretch', hide_index=True)
                
                st.divider()
                
                # Recovery trend chart: Selected Indicator vs Recovery Index
                st.markdown("#### Indicator vs Recovery Index Trend")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    trend_indicator = st.selectbox(
                        "Select Indicator to Compare",
                        sorted([i for i in df['Indicator'].dropna().unique() if pd.notna(i)]),
                        key="recovery_trend_indicator"
                    )
                
                # Calculate recovery index over time - USE FULL DATASET MIN DATE
                country_data = df[df['Country'] == recovery_country].copy()
                dataset_min_date = df['Time'].min()  # Get minimum date from full dataset
                country_data = country_data[
                    (country_data['Time'] >= dataset_min_date) &  # Changed from base_date to dataset_min_date
                    (country_data['Time'] <= recovery_end_date)
                ].sort_values('Time')
                
                # Get unique dates
                unique_dates = sorted(country_data['Time'].unique())
                
                recovery_trend = []
                for date in unique_dates:
                    idx, _ = calculate_recovery_index(
                        country_data,
                        base_date,
                        date
                    )
                    if idx is not None:
                        recovery_trend.append({'Time': date, 'Recovery_Index': idx})
                
                if recovery_trend:
                    recovery_trend_df = pd.DataFrame(recovery_trend)
                    
                    # Get indicator data
                    indicator_data = country_data[country_data['Indicator'] == trend_indicator].copy()
                    
                    if len(indicator_data) > 0:
                        # Create dual-axis chart
                        fig = go.Figure()
                        
                        # Recovery Index (primary y-axis)
                        fig.add_trace(go.Scatter(
                            x=recovery_trend_df['Time'],
                            y=recovery_trend_df['Recovery_Index'],
                            name='Recovery Index',
                            line=dict(color='#11998e', width=3),
                            yaxis='y'
                        ))
                        
                        # Selected Indicator (secondary y-axis)
                        fig.add_trace(go.Scatter(
                            x=indicator_data['Time'],
                            y=indicator_data['Amount'],
                            name=trend_indicator,
                            line=dict(color='#667eea', width=3),
                            yaxis='y2'
                        ))
                        
                        # Add baseline marker
                        if isinstance(base_date, pd.Timestamp):
                            base_date_display = base_date.to_pydatetime()
                        else:
                            base_date_display = base_date

                        fig.add_vline(
                            x=base_date_display,
                            line_dash="dash",
                            line_color="red"
                        )
                        
                        fig.update_layout(
                            title=f"{recovery_country}: {trend_indicator} vs Recovery Index",
                            xaxis=dict(title="Date"),
                            yaxis=dict(
                                title="Recovery Index",
                                tickfont=dict(color="#11998e")
                            ),
                            yaxis2=dict(
                                title=trend_indicator,
                                tickfont=dict(color="#667eea"),
                                overlaying='y',
                                side='right'
                            ),
                            height=600,
                            hovermode='x unified',
                            legend=dict(x=0.01, y=0.99)
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                        
                        # Generate AI insight for recovery
                        if api_key:
                            with st.spinner("🤖 Generating AI insight..."):
                                try:
                                    client = Groq(api_key=api_key)
                                    comp_summary = ", ".join([f"{k}: {v['recovery_pct']:.0f}% recovery" for k, v in recovery_components.items()])
                                    response = client.chat.completions.create(
                                        model="llama-3.3-70b-versatile",
                                        messages=[
                                            {"role": "system", "content": "You are an expert economist. Provide brief recovery analysis in 2-3 sentences."},
                                            {"role": "user", "content": f"Analyze {recovery_country}'s economic recovery from {base_period}. Overall recovery index: {recovery_index:.0f}/100. Component recoveries: {comp_summary}. What does this tell us about the recovery progress?"}
                                        ],
                                        max_tokens=200
                                    )
                                    insight = response.choices[0].message.content
                                    st.markdown(f'<div class="info-card">🤖 <strong>AI Insight:</strong> {insight}</div>', unsafe_allow_html=True)
                                except:
                                    pass
                    else:
                        st.warning(f"No data available for {trend_indicator} in the selected period.")
                else:
                    st.warning("Insufficient data to calculate recovery trend.")
                
            else:
                st.warning(f"Insufficient data for {recovery_country} in the selected period to calculate recovery index.")
        
        with tabs[5]:  # Risk Assessment
            st.markdown('<div class="section-header">Economic Risk Monitor</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-card">⚠️ <strong>Assess risks:</strong> Identify economic downturn risks based on GDP trends, inflation, and other key indicators.</div>', unsafe_allow_html=True)
            
            for country in selected:
                risk = predict_downturn(df_filtered, country)
                
                with st.expander(f"🌍 {country} Risk Assessment", expanded=(len(selected) <= 3)):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        risk_color = "🟢" if risk['risk_level']=='Low' else "🟡" if risk['risk_level']=='Medium' else "🔴"
                        st.metric("Risk Level", f"{risk_color} {risk['risk_level']}", f"{risk['risk_score']}/100")
                    
                    with col2:
                        st.markdown("**Risk Factors Identified:**")
                        if risk['risk_factors']:
                            for factor in risk['risk_factors']:
                                st.warning(f"⚠️ {factor}")
                        else:
                            st.success("✅ No major risk factors identified")
                    
                    # Generate AI insight for risk
                    if api_key and risk['risk_factors']:
                        with st.spinner("🤖 Generating AI insight..."):
                            try:
                                client = Groq(api_key=api_key)
                                factors_str = "; ".join(risk['risk_factors'])
                                response = client.chat.completions.create(
                                    model="llama-3.3-70b-versatile",
                                    messages=[
                                        {"role": "system", "content": "You are an expert economist. Provide brief risk analysis in 2-3 sentences."},
                                        {"role": "user", "content": f"Analyze {country}'s economic risk (level: {risk['risk_level']}, score: {risk['risk_score']}/100). Risk factors: {factors_str}. What actions or monitoring should be prioritized?"}
                                    ],
                                    max_tokens=200
                                )
                                insight = response.choices[0].message.content
                                st.markdown(f'<div class="info-card">🤖 <strong>AI Insight:</strong> {insight}</div>', unsafe_allow_html=True)
                            except:
                                pass
        
        with tabs[6]:  # Export
            st.markdown('<div class="section-header">Export Data & Reports</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-card">📥 <strong>Download your data:</strong> Export filtered data and generate comprehensive reports.</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📄 Raw Data Export")
                st.caption("Download your filtered dataset in various formats")
                
                csv = export_csv(df_filtered)
                st.download_button(
                    "📊 Download as CSV",
                    csv,
                    "economic_data.csv",
                    "text/csv",
                    width='stretch',
                    help="Download data in CSV format for Excel"
                )
                
                json_data = df_filtered.to_json(orient='records', date_format='iso')
                st.download_button(
                    "📋 Download as JSON",
                    json_data,
                    "economic_data.json",
                    "application/json",
                    width='stretch',
                    help="Download data in JSON format"
                )
            
            with col2:
                st.markdown("#### 📝 Generate Reports")
                st.caption("Create comprehensive text reports for selected countries")
                
                if st.button("📄 Generate Text Reports", width='stretch', type="primary"):
                    with st.spinner("📝 Generating reports..."):
                        reports = []
                        for country in selected:
                            c_data = df_filtered[df_filtered['Country'] == country]
                            score, comp = calculate_health_score(c_data, selected_date)
                            if score:
                                report = generate_report_text(country, score, comp, df_filtered)
                                reports.append(report)
                        
                        if reports:
                            full_report = "\n\n".join(reports)
                            st.download_button(
                                "💾 Download All Reports",
                                full_report,
                                f"economic_reports_{datetime.now().strftime('%Y%m%d')}.txt",
                                "text/plain",
                                width='stretch'
                            )
                            st.success(f"✅ Successfully generated {len(selected)} reports!")
                        else:
                            st.warning("⚠️ No reports could be generated with the current selection.")
            
            st.divider()
            
            st.markdown("#### 📊 Export Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🌍 Countries", len(selected))
            with col2:
                st.metric("📈 Records", f"{len(df_filtered):,}")
            with col3:
                st.metric("📊 Indicators", df_filtered['Indicator'].nunique())
            with col4:
                date_range = (df_filtered['Time'].max() - df_filtered['Time'].min()).days
                st.metric("📅 Days Covered", f"{date_range:,}")
    
    else:
        st.info("👆 **Get Started:** Select at least one country from the filters above to begin your analysis!")
        st.markdown("""
        ### 🎯 Quick Guide
        
        1. **Select Countries**: Choose one or more countries from the filter
        2. **Set Time Range**: Pick your analysis period
        3. **Explore Tabs**: Navigate through different analysis views
        4. **Get Insights**: Use the AI assistant for deeper understanding
        5. **Export Results**: Download data and reports when done
        """)

if __name__ == "__main__":
    main()