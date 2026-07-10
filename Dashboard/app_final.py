import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import random
import plotly.graph_objects as go


from pathlib import Path
import base64

# ==========================
# Project Paths
# ==========================
BASE_DIR = Path(__file__).parent

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="AI Space Mission Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)
# ==========================
# Sidebar State
# ==========================
if "show_sidebar" not in st.session_state:
    st.session_state.show_sidebar = True

# ==========================
# Load Background Image
# ==========================
image_path = BASE_DIR / "assets" / "satellite-with-earth-background.png"

with open(image_path, "rb") as f:
    encoded = base64.b64encode(f.read()).decode()


st.markdown(
    f"""
    <style>

    [data-testid="stAppViewContainer"] {{

        background:
        linear-gradient(
            rgba(5,10,25,0.75),
            rgba(5,10,25,0.75)
        ),
        url("data:image/png;base64,{encoded}");

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;

    }}

    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# Load CSS
# ==========================
css_path = BASE_DIR / "style.css"

with open(css_path) as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )


@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/cleaned_space_missions.csv"
    )

    df["Country"] = (
        df["Location"]
        .str.split(",")
        .str[-1]
        .str.strip()
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    df["Year"] = df["Date"].dt.year

    return df

df = load_data()

model = joblib.load(
    "model/mission_model.pkl"
)

encoders = joblib.load(
    "model/encoders.pkl"
)

if st.button("☰ Toggle Sidebar"):
    st.session_state.show_sidebar = not st.session_state.show_sidebar
    st.rerun()

if st.session_state.show_sidebar:

    st.sidebar.title("🚀 Mission Filters")

    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/d/d0/International_Space_Station.svg"
    )

    st.sidebar.markdown("## 🚀 Mission Control")

    selected_company = st.sidebar.selectbox(
        "Company",
        ["All"] + sorted(df["Company"].unique())
    )

    selected_status = st.sidebar.selectbox(
        "Rocket Status",
        ["All"] + sorted(df["RocketStatus"].unique())
    )

    selected_country = st.sidebar.selectbox(
        "Country",
        ["All"] + sorted(df["Country"].unique())
    )

else:
    selected_company = "All"
    selected_status = "All"
    selected_country = "All"

filtered_df = df.copy()

if selected_company != "All":
    filtered_df = filtered_df[
        filtered_df["Company"]
        == selected_company
    ]

if selected_status != "All":
    filtered_df = filtered_df[
        filtered_df["RocketStatus"]
        == selected_status
    ]

if selected_country != "All":
    filtered_df = filtered_df[
        filtered_df["Country"]
        == selected_country
    ]

total_missions = len(filtered_df)

success_count = len(
    filtered_df[
        filtered_df["MissionStatus"]
        == "Success"
    ]
)

success_rate = (
    success_count /
    total_missions * 100
) if total_missions > 0 else 0

avg_price = (
    filtered_df["Price"]
    .fillna(
        filtered_df["Price"].median()
    )
    .mean()
)

top_company = (
    filtered_df["Company"]
    .value_counts()
    .idxmax()
    if total_missions > 0
    else "N/A"
)

st.title(
    "🚀 AI Space Mission Intelligence Dashboard"
)

st.markdown("""
### Data Analytics + Machine Learning + Space Exploration
""")

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric(
        "🌍 Missions",
        total_missions
    )

with c2:
    st.metric(
        "📈 Success Rate",
        f"{success_rate:.2f}%"
    )

with c3:
    st.metric(
        "💰 Avg Cost",
        f"${avg_price:.2f}M"
    )

with c4:
    st.metric(
        "🏆 Top Company",
        top_company
    )

fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=success_rate,
        title={'text':"Mission Success"},
        gauge={'axis':{'range':[0,100]}}
    )
)

st.plotly_chart(fig)

st.markdown("---")

left,right = st.columns(2)

launch_trend = (
    filtered_df["Year"]
    .value_counts()
    .sort_index()
    .reset_index()
)

launch_trend.columns = [
    "Year",
    "Launches"
]

fig1 = px.line(
    launch_trend,
    x="Year",
    y="Launches",
    markers=True,
    title="Launch Trend"
)

fig1.update_layout(
    template="plotly_dark"
)

country_df = (
    filtered_df["Country"]
    .value_counts()
    .head(10)
    .reset_index()
)

country_df.columns = [
    "Country",
    "Launches"
]

fig2 = px.bar(
    country_df,
    x="Country",
    y="Launches",
    title="Top Countries"
)

fig2.update_layout(
    template="plotly_dark"
)

with left:
    st.plotly_chart(
        fig1,
        width="stretch"
    )

with right:
    st.plotly_chart(
        fig2,
        width="stretch"
    )

left2,right2 = st.columns(2)

status_df = (
    filtered_df["MissionStatus"]
    .value_counts()
    .reset_index()
)

status_df.columns = [
    "MissionStatus",
    "Count"
]

fig3 = px.pie(
    status_df,
    names="MissionStatus",
    values="Count",
    hole=0.4
)

fig3.update_layout(
    template="plotly_dark"
)

company_df = (
    filtered_df["Company"]
    .value_counts()
    .head(10)
    .reset_index()
)

company_df.columns = [
    "Company",
    "Launches"
]

fig4 = px.bar(
    company_df,
    x="Company",
    y="Launches",
    title="Top Companies"
)

fig4.update_layout(
    template="plotly_dark"
)

with left2:
    st.plotly_chart(
        fig3,
        width="stretch"
    )

with right2:
    st.plotly_chart(
        fig4,
        width="stretch"
    )

st.markdown("---")

st.header(
    "🤖 AI Mission Success Predictor"
)

col1,col2 = st.columns(2)

with col1:

    company = st.selectbox(
        "Company",
        sorted(df["Company"].unique())
    )

    country = st.selectbox(
        "Country",
        sorted(df["Country"].unique())
    )

with col2:

    rocket_status = st.selectbox(
        "Rocket Status",
        sorted(df["RocketStatus"].unique())
    )

    price = st.slider(
        "Mission Cost",
        0,
        500,
        50
    )



if st.button(
    "🚀 Predict Outcome"
):

    try:

        company_encoded = (
            encoders["Company"]
            .transform([company])[0]
        )

        country_encoded = (
            encoders["Country"]
            .transform([country])[0]
        )

        rocket_encoded = (
            encoders["RocketStatus"]
            .transform([rocket_status])[0]
        )

        sample = pd.DataFrame({

            "Company":
            [company_encoded],

            "RocketStatus":
            [rocket_encoded],

            "Country":
            [country_encoded],

            "Price":
            [price]

        })

        prediction = (
            model.predict(sample)[0]
        )

        confidence = (
            float(
                model
                .predict_proba(sample)[0]
                .max()
            ) * 100
        )

        status = (
            encoders["MissionStatus"]
            .inverse_transform(
                [prediction]
            )[0]
        )

        if status == "Success":

            st.success(
                f"🚀 {status}"
            )

        else:

            st.error(
                f"⚠ {status}"
            )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

    except Exception as e:

        st.error(
            f"{e}"
        )

st.markdown("---")

st.header(
    "🔍 AI Insights"
)

top_country = (
    filtered_df["Country"]
    .value_counts()
    .idxmax()
)

st.markdown(f"""
### Key Findings

- 🚀 Most Active Company: **{top_company}**
- 🌎 Most Active Country: **{top_country}**
- 📈 Success Rate: **{success_rate:.2f}%**
- 💰 Average Cost: **${avg_price:.2f}M**
""")

csv = filtered_df.to_csv(
    index=False
)

st.download_button(
    "📥 Download Dataset",
    csv,
    "space_missions.csv",
    "text/csv"
)

space_facts = [

"🚀 Apollo 11 landed on the Moon in 1969.",

"🪐 Saturn has more than 140 moons.",

"☄ Voyager 1 is the farthest human-made object.",

"🌌 Milky Way contains billions of stars.",

"🌍 Earth travels around the Sun at 107,000 km/h."
]

st.markdown("---")

st.subheader(
    "🌌 Space Fact"
)

st.info(
    random.choice(space_facts)
)

st.markdown("---")

st.markdown("""
<center>

Developed by Janvi Tailor 🚀

</center>
""",
unsafe_allow_html=True)