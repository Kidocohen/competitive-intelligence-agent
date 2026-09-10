import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ייבוא הכלים שהוגדרו
from tools import system_tools

load_dotenv()



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_safe: bool
    iteration_count: int



llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


guardrail_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


llm_with_tools = llm.bind_tools(system_tools)



#מימוש חמשת הצמתים

# 1

def input_guardrail_node(state: AgentState):
    """
    בודק האם הקלט קשור למודיעין עסקי, חברות, מתחרים או מידע ארגוני פנימי.
    מונע פניות ל-Qdrant ול-n8n בנושאים שאינם רלוונטיים.
    """
    last_message = state["messages"][-1].content.strip()

    if not last_message:
        return {
            "messages": [AIMessage(content="[Scope Notice] Query cannot be empty.")],
            "is_safe": False
        }


    forbidden_keywords = ["ignore previous instructions", "drop table", "bypass system prompt"]
    if any(keyword in last_message.lower() for keyword in forbidden_keywords):
        return {
            "messages": [AIMessage(content="[Security Alert] Prompt injection attempt detected. Request halted.")],
            "is_safe": False
        }


    relevance_prompt = (
        "You are an intake filter for an Elite Competitive Intelligence and Corporate Research Agent.\n"
        "Your task is to determine whether the user query belongs to the domain of: "
        "business intelligence, market research, corporate strategies, company finances, target competitors, "
        "industry trends, products, or internal corporate policies and operations.\n\n"
        f"User Query: \"{last_message}\"\n\n"
        "Decision Rule:\n"
        "- If the query is related to business, corporate data, companies, competitors, market analysis, or internal organization queries, answer 'RELEVANT'.\n"
        "- If the query is off-topic (e.g., general pet care/entry, cooking recipes, weather, personal advice, casual chit-chat, pop culture trivia), answer 'OFF_TOPIC'.\n"
        "Respond ONLY with one word: RELEVANT or OFF_TOPIC."
    )

    verdict = guardrail_llm.invoke([HumanMessage(content=relevance_prompt)]).content.strip().upper()

    if "OFF_TOPIC" in verdict:
        out_of_scope_reply = (
            "I am a specialized Competitive & Market Intelligence Agent. "
            "Your query is outside the operational scope of corporate research, competitor analysis, "
            "and business intelligence. Please provide a business or market-related directive."
        )
        return {
            "messages": [AIMessage(content=out_of_scope_reply)],
            "is_safe": False
        }

    return {"is_safe": True, "iteration_count": 0}


# 2
def research_planner_node(state: AgentState):
    """בונה הנחיית מערכת ומגדיר את אסטרטגיית איסוף המודיעין."""
    planner_prompt = SystemMessage(
        content=(
            "You are an Elite Competitive Intelligence Agent. "
            "Your objective is to conduct comprehensive market research on target competitors. "
            "Follow these guidelines:\n"
            "1. Query internal intelligence first using 'search_internal_intelligence'.\n"
            "2. If internal records are incomplete or external verification is required, fetch live signals using 'fetch_external_market_data'.\n"
            "3. Synthesize your findings into a clear strategic dossier with recommendations grounded strictly in retrieved facts."
        )
    )
    return {"messages": [planner_prompt]}


#3
def agent_reasoning_node(state: AgentState):
    """מפעיל את מודל השפה לקבלת החלטה: קריאה לכלי או ניסוח מסקנה."""
    current_count = state.get("iteration_count", 0) + 1
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response], "iteration_count": current_count}


# 4
tool_node = ToolNode(system_tools)


# 5
def output_guardrail_node(state: AgentState):
    """
    מוודא שהתשובה הסופית:
    1. אינה ריקה ובעלת עומק ניתוחי מספק.
    2. אינה כוללת שגיאות מערכת גולמיות או תגיות שבורות.
    3. מבוססת על המידע שנאסף מהכלים (Hallucination & Groundedness check).
    """
    last_message = state["messages"][-1]
    content = last_message.content


    if not content or len(content.strip()) < 50:
        warning_suffix = "\n\n[Warning: Generated report lacks sufficient analytical depth.]"
        return {"messages": [AIMessage(content=content + warning_suffix)]}


    raw_artifacts = ["{{", "}}", "undefined", "[object Object]"]
    if any(artifact in content for artifact in raw_artifacts):
        cleaned_content = content
        for artifact in raw_artifacts:
            cleaned_content = cleaned_content.replace(artifact, "")
        content = cleaned_content + "\n\n[Notice: Output sanitized by Output Guardrail.]"


    has_tool_context = any(
        getattr(m, "type", "") == "tool" or hasattr(m, "tool_call_id")
        for m in state["messages"]
    )


    if has_tool_context:
        evidence_indicators = [
            "market", "intelligence", "revenue", "source",
            "financial", "trend", "risk", "internal", "external", "data", "report"
        ]
        has_evidence = any(word in content.lower() for word in evidence_indicators)
        if not has_evidence:
            content += "\n\n[Guardrail Advisory: Report may lack grounded factual anchors from retrieved tool data.]"

    return {"messages": [AIMessage(content=content)]}


# conditional edges

def route_after_input_guardrail(state: AgentState):
    """אם הקלט סווג כ-OFF_TOPIC או לא בטוח, מסיים מיד."""
    if not state.get("is_safe", True):
        return END
    return "research_planner"


def route_agent_decision(state: AgentState):
    """בודק אם הסוכן ביקש לקרוא לכלי או שסיים את מחקרו."""
    last_message = state["messages"][-1]

    if state.get("iteration_count", 0) >= 5:
        return "output_guardrail"

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_execution"

    return "output_guardrail"



# הגרף


workflow = StateGraph(AgentState)

workflow.add_node("input_guardrail", input_guardrail_node)
workflow.add_node("research_planner", research_planner_node)
workflow.add_node("agent_reasoning", agent_reasoning_node)
workflow.add_node("tool_execution", tool_node)
workflow.add_node("output_guardrail", output_guardrail_node)

workflow.set_entry_point("input_guardrail")

workflow.add_conditional_edges(
    "input_guardrail",
    route_after_input_guardrail,
    {
        "research_planner": "research_planner",
        END: END
    }
)

workflow.add_edge("research_planner", "agent_reasoning")

workflow.add_conditional_edges(
    "agent_reasoning",
    route_agent_decision,
    {
        "tool_execution": "tool_execution",
        "output_guardrail": "output_guardrail"
    }
)

workflow.add_edge("tool_execution", "agent_reasoning")
workflow.add_edge("output_guardrail", END)

app_graph = workflow.compile()



# בדיקה והפקת ויזואליזציה

if __name__ == "__main__":
    print("=== Testing Competitive Intelligence LangGraph ===")

    try:
        docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
        img_path = os.path.join(docs_dir, "langgraph_visualization.png")
        png_data = app_graph.get_graph().draw_mermaid_png()
        with open(img_path, "wb") as f:
            f.write(png_data)
        print(f"[Visualization] Graph structure saved to: {img_path}")
    except Exception as e:
        print(f"[Visualization Notice] Could not render PNG: {e}")

    # בדיקה מהירה של סינון שאילתה מחוץ לתחום
    test_input = {
        "messages": [HumanMessage(content="which dogs are allowed to enter your site?")]
    }
    print("\n[Running Execution] Invoking graph with off-topic query...")
    final_output = app_graph.invoke(test_input)

    print("\n=== Final Response Generated ===")
    print(final_output["messages"][-1].content)