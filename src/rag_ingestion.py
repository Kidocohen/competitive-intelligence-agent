import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# טעינת משתני הסביבה
load_dotenv()


def build_cloud_vector_store():

    openai_api_key = os.getenv("OPENAI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not openai_api_key or not qdrant_url or not qdrant_api_key:
        raise ValueError("[ERROR] Missing required keys in .env: OPENAI_API_KEY, QDRANT_URL, or QDRANT_API_KEY")


    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "internal_research_report.txt")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[ERROR] Source document not found at: {data_path}")

    print(f"[RAG] Loading intelligence document from: {data_path}")
    loader = TextLoader(data_path, encoding="utf-8")
    documents = loader.load()


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[RAG] Document partitioned into {len(chunks)} contextual chunks.")


    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )


    collection_name = "competitive_intelligence"
    print(f"[RAG] Connecting to Qdrant at: {qdrant_url}")

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key
    )


    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=1536,
            distance=qmodels.Distance.COSINE
        )
    )


    print(f"[RAG] Generating embeddings and uploading to '{collection_name}'...")
    points = []

    for idx, chunk in enumerate(chunks):
        content = chunk.page_content
        vector = embeddings.embed_query(content)


        target_company = "AlphaCorp" if "alphacorp" in content.lower() else "General Corporate"

        point = qmodels.PointStruct(
            id=idx,
            vector=vector,
            payload={
                "text": content,
                "page_content": content,
                "source": "internal_research_report.txt",
                "company": target_company
            }
        )
        points.append(point)

    client.upsert(
        collection_name=collection_name,
        points=points
    )

    print(f"[RAG] Successfully ingested {len(points)} vector records into Qdrant collection '{collection_name}'.")


if __name__ == "__main__":
    build_cloud_vector_store()