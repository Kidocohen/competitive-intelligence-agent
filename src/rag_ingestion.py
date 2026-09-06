import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# טעינת משתני הסביבה מתוך קובץ .env
load_dotenv()

def build_cloud_vector_store():
    # שלב 1: אימות קיומם של המפתחות הנדרשים
    openai_api_key = os.getenv("OPENAI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not openai_api_key or not qdrant_url or not qdrant_api_key:
        raise ValueError("[ERROR] Missing required keys in .env: OPENAI_API_KEY, QDRANT_URL, or QDRANT_API_KEY")

    # שלב 2: הגדרת נתיב הקובץ וטעינת מסמך המקור
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "internal_research_report.txt")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[ERROR] Source document not found at: {data_path}")

    print(f"[RAG] Loading intelligence document from: {data_path}")
    loader = TextLoader(data_path, encoding="utf-8")
    documents = loader.load()

    # שלב 3: אסטרטגיית חלוקה למקטעים (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[RAG] Document partitioned into {len(chunks)} contextual chunks.")

    # שלב 4: אתחול מודל הטמעה (Embedding) חסכוני
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )

    # שלב 5: העלאת הווקטורים לקלאסטר בענן של Qdrant
    collection_name = "competitive_intelligence"
    print(f"[RAG] Connecting to Qdrant Cloud at: {qdrant_url}")

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key
    )

    print(f"[RAG] Uploading vector embeddings to collection '{collection_name}'...")
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        force_recreate=True
    )

    print("[RAG] Vector ingestion to Qdrant Cloud completed successfully!")
    return vector_store

if __name__ == "__main__":
    build_cloud_vector_store()