import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Competitive Intelligence Agent",
    page_icon="🔍",
    layout="wide"
)

# כתובת ה-API: תמיכה בשם השירות בדוקר עם נפילה ל-localhost
API_URL = os.getenv("API_URL", "http://competitive_intelligence_agent:5000/api/research")

st.title("Competitive Intelligence Agent")
st.caption("LangGraph Multi-Agent Engine with Vector Search & REST API")

# שדה הזנת השאילתה
user_prompt = st.text_area(
    "Enter Intelligence Directive:",
    value="What are the primary weaknesses and deployment risks of EpsilonAutomate?",
    height=120
)

# כפתור הפעלה
if st.button("Run Analysis", type="primary"):
    if not user_prompt.strip():
        st.warning("Please enter a valid intelligence directive.")
    else:
        with st.spinner("Agent is gathering intelligence and synthesizing report..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"query": user_prompt},
                    timeout=120
                )

                if response.status_code == 200:
                    data = response.json()
                    st.success("Analysis Complete!")

                    # הצגת התוצאה הסופית של הסוכן
                    final_report = data.get("report") or data.get("response") or data.get("result") or str(data)
                    st.markdown("### Strategic Intelligence Dossier")
                    st.markdown(final_report)

                    # פירוט שלבים / נתונים נוספים אם קיימים בתשובה
                    if "sources" in data or "retrieved_chunks" in data:
                        with st.expander("Retrieved Context & Sources"):
                            st.write(data.get("sources") or data.get("retrieved_chunks"))

                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Connection Failed: {e}")