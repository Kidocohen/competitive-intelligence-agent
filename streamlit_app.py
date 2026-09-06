import os
import requests
import streamlit as st

st.set_page_config(page_title="Competitive Intelligence Agent", layout="wide")

st.title("Competitive Intelligence Agent")
st.caption("LangGraph Multi-Agent Engine with Vector Search & REST API")

user_query = st.text_area(
    "Enter Intelligence Directive:",
    placeholder="e.g., What are the main risks associated with BetaTech?"
)

if st.button("Run Analysis"):
    if not user_query.strip():
        st.warning("Please enter a directive.")
    else:
        with st.spinner("Analyzing intelligence data..."):
            try:
                api_host = os.getenv("API_HOST", "localhost")
                api_url = f"http://{api_host}:5000/api/research"

                response = requests.post(
                    api_url,
                    json={"query": user_query}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success("Analysis Complete")
                    st.markdown("### Findings Report")
                    st.markdown(data.get("report", "No report text received."))
                else:
                    st.error(f"API Error: Status {response.status_code}")
            except Exception as e:
                st.error(f"Connection Failed: {str(e)}")