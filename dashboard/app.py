from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
data_path = ROOT / "results" / "model_ready_user_engagement_data.csv"

st.set_page_config(page_title="Mobile App Engagement Analytics", layout="wide")
st.title("Mobile App User Engagement Analytics")
st.caption("Seven-day activity prediction and engagement analysis")

if not data_path.exists():
    st.error("Run `python run_project.py` first to generate the model-ready dataset.")
    st.stop()

df = pd.read_csv(data_path)

total_users = len(df)
active = int(df["active_next_7_days"].sum())
inactive = int((df["active_next_7_days"] == 0).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Users", f"{total_users:,}")
c2.metric("Active Next 7 Days", f"{active:,}")
c3.metric("Not Active", f"{inactive:,}")
c4.metric("Avg Events / User", f"{df['total_events'].mean():.2f}")

st.subheader("7-Day Activity")
st.bar_chart(
    df["active_next_7_days"].value_counts().rename(index={0: "Not Active", 1: "Active"})
)

st.subheader("User Engagement Distribution")
st.bar_chart(df["total_events"].value_counts().sort_index())

st.subheader("Operating System")
st.bar_chart(df["device_os"].value_counts())

st.subheader("Top Countries")
st.bar_chart(df["location_country"].value_counts().head(10))
