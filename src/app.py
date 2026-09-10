import os
import sys
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain_core.messages import HumanMessage


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph import app_graph

load_dotenv()

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "Competitive Intelligence Agent API"}), 200


@app.route("/api/research", methods=["POST"])
def run_research():
    """
    מקבל שאילתת מחקר ומפעיל את סוכן ה-LangGraph.
    מבנה קלט מצופה:
    {
        "query": "What are the vulnerabilities of AlphaCorp?"
    }
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field in request body"}), 400

    user_query = data["query"]

    try:
        # הרצת הגרף
        initial_input = {"messages": [HumanMessage(content=user_query)]}
        result = app_graph.invoke(initial_input)

        final_message = result["messages"][-1].content
        is_safe = result.get("is_safe", True)

        return jsonify({
            "status": "success",
            "query": user_query,
            "report": final_message,
            "is_safe": is_safe
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Execution failure: {str(e)}"
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"Starting API Server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)