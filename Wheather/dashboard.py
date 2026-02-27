import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from streamlit_folium import st_folium
import folium
from streamlit_lottie import st_lottie
import requests
from weather_pipeline import run_pipeline, DATA_FILE

# Page Config
st.set_page_config(
    page_title="WeatherAI Command Center",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie Assets
LOTTIE_SUN = "https://assets9.lottiefiles.com/packages/lf20_xlky4kvh.json"
LOTTIE_RAIN = "https://assets9.lottiefiles.com/packages/lf20_k6wspx1i.json"
LOTTIE_CLOUD = "https://assets9.lottiefiles.com/packages/lf20_k2caxh5a.json"
LOTTIE_SNOW = "https://assets9.lottiefiles.com/packages/lf20_6aYlH8.json"
LOTTIE_STORM = "https://assets9.lottiefiles.com/packages/lf20_kuw7q5.json"

def get_lottie_for_weather(desc):
    desc = str(desc).lower()
    if 'rain' in desc or 'drizzle' in desc: return LOTTIE_RAIN
    if 'cloud' in desc or 'overcast' in desc: return LOTTIE_CLOUD
    if 'snow' in desc: return LOTTIE_SNOW
    if 'storm' in desc or 'thunder' in desc: return LOTTIE_STORM
    return LOTTIE_SUN

def generate_ai_insight(row):
    """Simple rule-based 'AI' insight generator"""
    temp = row['temperature_c']
    hum = row['humidity']
    wind = row.get('wind_speed', 0)
    desc = row['description']
    
    insight = f"Currently experiencing **{desc}**."
    
    if temp > 30:
        insight += " High thermal load detected. Stay hydrated."
    elif temp < 5:
        insight += " Sub-optimal thermal conditions. Thermal insulation recommended."
    
    if hum > 80:
        insight += " Atmospheric saturation levels high. Expect precipitation risk."
    
    if pd.notna(wind) and wind > 5:
        insight += f" Significant air displacement recorded ({wind} m/s)."
        
    return insight

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    .metric-card {
        background-color: #1a1c24;
        border: 1px solid #30333d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .big-stat {
        font-size: 3em;
        font-weight: bold;
        color: #4DB6AC;
        margin: 0;
    }
    .stat-label {
        color: #B0BEC5;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
c1, c2 = st.columns([1, 6])
with c1:
    try:
        logo_lottie = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_bwmvzkb9.json")
        if logo_lottie:
            st_lottie(logo_lottie, height=80, key="logo")
        else:
            st.markdown("## 🌪️") # Fallback icon
    except Exception:
        st.markdown("## 🌪️")
with c2:
    st.markdown("# 🌪️ WeatherAI Command Center")
    st.markdown("### Real-Time Atmospheric Intelligence System")

# --- Sidebar ---
st.sidebar.markdown("## 📡 Mission Control")

# Initialize session state for city
if 'city' not in st.session_state:
    st.session_state.city = "London"

# Input for Manual City Entry
st.sidebar.markdown("### 🎯 Target Selection")
city_input = st.sidebar.text_input("Enter City", st.session_state.city).strip()

if city_input and city_input != st.session_state.city:
    st.session_state.city = city_input
    with st.spinner(f"🚀 Re-aligning satellites to {city_input}..."):
        new_city = run_pipeline(city=city_input)
        if new_city: st.session_state.city = new_city
    st.rerun()

st.sidebar.divider()
refresh_rate = st.sidebar.slider("⏱️ Scan Interval (s)", 5, 60, 10)
auto_refresh = st.sidebar.toggle("⚡ Auto-Scan Sequence", value=False)
history_limit = st.sidebar.select_slider("📅 Temporal Window", options=["Last 10", "Last 50", "Last 100", "All Time"], value="Last 50")

st.sidebar.divider()
if st.sidebar.button("🛰️ Acquire Immediate Data", type="primary", use_container_width=True):
    with st.spinner(f"Acquiring telemetry for {st.session_state.city}..."):
        run_pipeline(city=st.session_state.city)
    st.success(f"Packet received: {st.session_state.city}")

# Map
    # Create a map centered on the last known location or London
    start_lat, start_lon = 51.5074, -0.1278
    
    cities_data = []
    try:
        df = pd.read_csv(DATA_FILE)
        if 'lat' in df.columns and 'lon' in df.columns:
            # Get latest unique location for each city
            cities_data = df.groupby('city').last().reset_index()
            
            # Center on current selected city
            current_city_data = cities_data[cities_data['city'].str.lower() == st.session_state.city.lower()]
            if not current_city_data.empty and pd.notna(current_city_data['lat'].values[0]):
                start_lat = current_city_data['lat'].values[0]
                start_lon = current_city_data['lon'].values[0]
    except: pass

    m = folium.Map(location=[start_lat, start_lon], zoom_start=4, tiles="CartoDB dark_matter")
    
    # Add markers for all known cities
    for _, row in cities_data.iterrows():
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            folium.Marker(
                [row['lat'], row['lon']], 
                popup=f"{row['city']}: {row['temperature_c']}°C",
                tooltip=row['city'],
                icon=folium.Icon(color='blue' if row['city'].lower() == st.session_state.city.lower() else 'gray', icon='cloud')
            ).add_to(m)

    m.add_child(folium.LatLngPopup())
    map_data = st_folium(m, height=400, use_container_width=True)

    if map_data and map_data.get("last_clicked"):
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        key = f"{lat}_{lon}"
        if 'last_click_key' not in st.session_state or st.session_state.last_click_key != key:
            st.session_state.last_click_key = key
            st.toast(f"📍 Coordinates Locked: {lat:.2f}, {lon:.2f}")
            with st.spinner("Triangulating..."):
                found = run_pipeline(city=st.session_state.city, lat=lat, lon=lon)
                if found:
                    st.session_state.city = found
                    time.sleep(1)
                    st.rerun()

# --- Main Logic ---
city = st.session_state.city
if auto_refresh:
    time.sleep(refresh_rate)
    run_pipeline(city=city)
    st.rerun()

try:
    df = pd.read_csv(DATA_FILE)
    city_df = df[df['city'].str.lower() == city.lower()].copy() # explicit copy
    
    if not city_df.empty:
        city_df.loc[:, 'timestamp'] = pd.to_datetime(city_df['timestamp'])
        
        # History filter
        if "10" in history_limit: ddf = city_df.tail(10)
        elif "50" in history_limit: ddf = city_df.tail(50)
        elif "100" in history_limit: ddf = city_df.tail(100)
        else: ddf = city_df
        
        latest = ddf.iloc[-1]
        
        # --- Top Section: Animation + Stats ---
        col_main_1, col_main_2 = st.columns([1, 2])
        
        with col_main_1:
            lottie_url = get_lottie_for_weather(latest['description'])
            lottie_json = load_lottieurl(lottie_url)
            if lottie_json:
                st_lottie(lottie_json, height=200, key="weather_anim")
            st.caption(f"📍 {latest['city']} | {latest['timestamp']}")
            
            # AI Insight Box
            st.info(f"🤖 **AI Analysis**: {generate_ai_insight(latest)}")

        with col_main_2:
            # Custom Metric Cards Grid
            mk1, mk2, mk3 = st.columns(3)
            with mk1:
                st.markdown(f"""<div class="metric-card"><div class="stat-label">Temperature</div><div class="big-stat">{latest['temperature_c']}°C</div></div>""", unsafe_allow_html=True)
            with mk2:
                st.markdown(f"""<div class="metric-card"><div class="stat-label">Humidity</div><div class="big-stat">{latest['humidity']}%</div></div>""", unsafe_allow_html=True)
            with mk3:
                pressure = latest['pressure']
                st.markdown(f"""<div class="metric-card"><div class="stat-label">Pressure</div><div class="big-stat">{pressure} <span style="font-size:0.4em">hPa</span></div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            mk4, mk5, mk6 = st.columns(3)
            with mk4:
                wind_spd = latest.get('wind_speed', 'N/A')
                st.markdown(f"""<div class="metric-card"><div class="stat-label">Wind Speed</div><div class="big-stat">{wind_spd} <span style="font-size:0.4em">m/s</span></div></div>""", unsafe_allow_html=True)
            with mk5:
                # Calculate simple trend
                if len(ddf) > 1:
                    delta = latest['temperature_c'] - ddf.iloc[-2]['temperature_c']
                    trend_symbol = "↗️" if delta > 0 else "↘️" if delta < 0 else "➡️"
                    trend_val = f"{abs(delta):.2f}"
                else:
                    trend_symbol = "➡️"
                    trend_val = "0.00"
                st.markdown(f"""<div class="metric-card"><div class="stat-label">Thermal Trend</div><div class="big-stat">{trend_symbol} {trend_val}</div></div>""", unsafe_allow_html=True)
            with mk6:
                st.markdown(f"""<div class="metric-card"><div class="stat-label">Data Points</div><div class="big-stat">{len(city_df)}</div></div>""", unsafe_allow_html=True)

        st.divider()

        # --- Advanced Visualizations Tabs ---
        tab1, tab2, tab3 = st.tabs(["📊 Telemetry Trends", "🧭 Wind & Compass", "💾 Data Logs"])

        with tab1:
            st.markdown("#### Thermal & Atmospheric Correlation")
            fig = go.Figure()
            # Gradient Area for Temp
            fig.add_trace(go.Scatter(
                x=ddf['timestamp'], y=ddf['temperature_c'],
                fill='tozeroy', mode='lines+markers', name='Temp (°C)',
                line=dict(color='#00E5FF', width=3),
                fillcolor='rgba(0, 229, 255, 0.1)'
            ))
            # Dashed line for Humidity
            fig.add_trace(go.Scatter(
                x=ddf['timestamp'], y=ddf['humidity'],
                mode='lines', name='Humidity (%)',
                line=dict(color='#76FF03', width=2, dash='dash'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#B0BEC5'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#263238'),
                yaxis2=dict(overlaying='y', side='right', showgrid=False),
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            c_wind, c_gauge = st.columns(2)
            with c_wind:
                st.markdown("#### Wind Vector Analysis")
                if pd.notna(latest.get('wind_speed')) and pd.notna(latest.get('wind_deg')):
                    # Polar Chart for Wind
                    fig_wind = go.Figure(go.Barpolar(
                        r=[latest['wind_speed']],
                        theta=[latest['wind_deg']],
                        width=[30], # sector width
                        marker_color=["#FF4081"],
                        marker_line_color="black",
                        marker_line_width=2,
                        opacity=0.8
                    ))
                    fig_wind.update_layout(
                        template='plotly_dark',
                        polar=dict(
                            radialaxis=dict(range=[0, max(10, latest['wind_speed']+5)], showticklabels=True, ticks=''),
                            angularaxis=dict(showticklabels=True, direction='clockwise')
                        ),
                        margin=dict(l=20, r=20, t=20, b=20),
                        height=300
                    )
                    st.plotly_chart(fig_wind, use_container_width=True)
                else:
                    st.info("Wind telemetry unavailable for this vector.")

            with c_gauge:
                st.markdown("#### Barometric Pressure")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=latest['pressure'],
                    delta={'reference': 1013, 'increasing':{'color':'#FF5252'}, 'decreasing':{'color':'#00E676'}},
                    gauge={
                        'axis': {'range': [950, 1060], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#448AFF"},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "#37474F",
                        'steps': [
                            {'range': [950, 1000], 'color': '#263238'},
                            {'range': [1000, 1060], 'color': '#37474F'}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_gauge, use_container_width=True)

        with tab3:
            st.markdown("#### Raw Data Logs")
            st.dataframe(
                ddf.sort_values(by='timestamp', ascending=False)
                .style.background_gradient(cmap="viridis", subset=['temperature_c', 'humidity', 'wind_speed']),
                use_container_width=True
            )

    else:
        st.warning("No signal detected. Initiate data acquisition.")

except Exception as e:
    st.error(f"System Malfunction: {e}")
