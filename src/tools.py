import os
import requests
from typing import Optional
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# הגדרות תקשורת מול n8n ו-Qdrant
N8N_WEBHOOK_URL = os.getenv(
    "N8N_WEBHOOK_URL",
    "http://n8n_local:5678/webhook/market-intelligence"
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "competitive_intelligence"

# אתחול קליינטים מרכזיים (Singleton) לחיסכון במשאבים
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


@tool
def search_internal_intelligence(query: str, company_filter: Optional[str] = None) -> str:
    """
    Search the internal competitive intelligence repository in Qdrant vector database.
    Use this tool FIRST to retrieve proprietary corporate dossiers, SWOT analyses, and confidential strategy notes.
    Optionally filter results by target company name.
    """
    try:
        query_vector = embeddings_model.embed_query(query)

        # תמיכה בסינון מטא-דאטה מתקדם לפי סעיף 4.2
        query_filter = None
        if company_filter:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="company",
                        match=qmodels.MatchValue(value=company_filter.strip().title())
                    )
                ]
            )

        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=3
        )

        if not results:
            return "No matching internal intelligence documents found in Qdrant repository."

        retrieved_texts = []
        for res in results:
            text = res.payload.get("text", "")
            source = res.payload.get("source", "Internal Intelligence DB")
            company = res.payload.get("company", "General Corporate")
            retrieved_texts.append(f"[Source: {source} | Entity: {company}]\n{text}")

        return "\n\n---\n\n".join(retrieved_texts)

    except Exception as e:
        return f"Error querying Qdrant internal intelligence: {str(e)}"


@tool
def fetch_external_market_data(company: str, query: str = "Latest market intelligence and corporate updates") -> str:
    """
    Trigger external market intelligence gathering via n8n automation webhook.
    Use this tool when internal intelligence is insufficient, or when fresh real-time web search and live market signals are required.
    """
    payload = {
        "company": company,
        "query": query
    }
    try:
        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=15)

        if response.status_code == 200:
            if not response.text.strip():
                return f"n8n Market Feed [Status: SUCCESS]: Webhook processed successfully for {company}."

            try:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    data = data[0]

                alert = data.get("alert", "External alert dispatched.")
                status = data.get("status", "SUCCESS")
                score = data.get("sentiment_score", "N/A")

                return f"n8n Live Intelligence [Status: {status} | Sentiment Score: {score}]:\n{alert}"

            except Exception:
                return f"n8n Market Feed [Raw Output]: {response.text}"

        return f"External n8n service returned HTTP {response.status_code}: {response.text}"

    except Exception as e:
        return f"Failed to contact external n8n automation service: {str(e)}"


# רשימת הכלים לחשיפה עבור סוכן ה-LangGraph
system_tools = [search_internal_intelligence, fetch_external_market_data]