import os
import requests
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.tools import tool

# טעינת משתני סביבה
load_dotenv()

# ==========================================
# 1. הגדרות מסד נתונים וקטורי (Qdrant Cloud)
# ==========================================
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key
)

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="competitive_intelligence",
    embedding=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# ==========================================
# 2. כלי שליפת מידע פנימי (RAG Tool)
# ==========================================
@tool
def search_internal_intelligence(query: str) -> str:
    """
    Search and retrieve confidential internal strategic intelligence documents,
    competitor benchmarks, financial targets, and internal directives.
    Input should be a targeted search query string.
    """
    try:
        docs = retriever.invoke(query)
        if not docs:
            return "No matching internal intelligence documents found."

        results = []
        for i, doc in enumerate(docs, start=1):
            results.append(f"--- Document Excerpt {i} ---\n{doc.page_content.strip()}")
        return "\n\n".join(results)
    except Exception as e:
        return f"Error retrieving internal intelligence: {str(e)}"


# ==========================================
# 3. כלי איסוף מידע חיצוני מ-n8n (External Fetcher Tool)
# ==========================================
@tool
def fetch_external_market_data(competitor_name: str) -> str:
    """
    Fetch live external market intelligence, recent public pricing changes,
    and scraping feeds regarding a competitor via n8n automation webhook.
    Input should be the competitor name (e.g., 'AlphaCorp', 'BetaTech').
    """
    webhook_url = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/competitive-intel")
    payload = {
        "competitor": competitor_name,
        "action": "fetch_market_intelligence"
    }

    try:
        # פנייה לשרת האוטומציה ב-n8n עם הגדרת Timeout של 5 שניות
        response = requests.post(webhook_url, json=payload, timeout=5)
        if response.status_code == 200:
            return f"Live Market Data from n8n: {response.text}"
        else:
            return f"n8n Webhook returned status {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        # מנגנון שרידות (Fallback): אם n8n טרם הופעל, מחזירים מענה מבוקר כדי שהסוכן לא יקרוס
        return (
            f"[Connection Notice] Unable to reach n8n webhook at {webhook_url}. "
            f"Simulated live update: '{competitor_name}' recently ran a marketing push on enterprise cloud features."
        )


# רשימת כל הכלים לייצוא אל סוכן ה-LangGraph
system_tools = [search_internal_intelligence, fetch_external_market_data]

# בדיקה עצמאית של שני הכלים
if __name__ == "__main__":
    print("=== Testing Tool 1: Internal RAG ===")
    rag_res = search_internal_intelligence.invoke({"query": "AlphaCorp vulnerabilities"})
    print(rag_res[:200] + "...\n")

    print("=== Testing Tool 2: n8n External Fetcher ===")
    n8n_res = fetch_external_market_data.invoke({"competitor_name": "AlphaCorp"})
    print(n8n_res)