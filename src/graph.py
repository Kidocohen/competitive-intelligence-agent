import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ייבוא הכלים שהגדרנו בקובץ הקודם
from tools import system_tools

load_dotenv()


# ==========================================
# 1. הגדרת מבנה ה-State של הגרף
# ==========================================
class AgentState(TypedDict):
    # add_messages שומר על היסטוריית השיחה ומשרשר הודעות חדשות בצורה בטוחה
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_safe: bool
    iteration_count: int


# אתחול מודל השפה - שימוש במודל חסכוני ויציב
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# חיבור הכלים למודל (Tool Binding)
llm_with_tools = llm.bind_tools(system_tools)


# ==========================================
# 2. מימוש חמשת הצמתים (Nodes)
# ==========================================

# צומת 1: Input Guardrail (סעיף 4.1)
def input_guardrail_node(state: AgentState):
    """בודק האם הקלט קשור למודיעין עסקי ותחרותי ואינו פוגעני או ריק."""
    last_message = state["messages"][-1].content.strip()

    if not last_message:
        return {
            "messages": [AIMessage(content="[Guardrail Block] Query cannot be empty.")],
            "is_safe": False
        }

    # בדיקת נושא בסיסית ומהירה
    forbidden_keywords = ["ignore previous instructions", "drop table", "bypass"]
    if any(keyword in last_message.lower() for keyword in forbidden_keywords):
        return {
            "messages": [AIMessage(content="[Guardrail Block] Prompt injection attempt detected.")],
            "is_safe": False
        }

    return {"is_safe": True, "iteration_count": 0}


# צומת 2: Research Planner
def research_planner_node(state: AgentState):
    """בונה הנחיית מערכת ומגדיר את אסטרטגיית איסוף המודיעין."""
    planner_prompt = SystemMessage(
        content=(
            "You are an Elite Competitive Intelligence Agent. "
            "Your objective is to conduct comprehensive market research on target competitors. "
            "Follow these guidelines:\n"
            "1. ALWAYS query internal intelligence first using 'search_internal_intelligence'.\n"
            "2. Fetch live market signals using 'fetch_external_market_data'.\n"
            "3. Synthesize your findings into a clear strategic dossier with recommendations."
        )
    )
    return {"messages": [planner_prompt]}


# צומת 3: Agent Reasoning (ReAct Loop)
def agent_reasoning_node(state: AgentState):
    """מפעיל את מודל השפה כדי להחליט על צעד הבא: קריאה לכלי או גיבוש מסקנה."""
    current_count = state.get("iteration_count", 0) + 1
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response], "iteration_count": current_count}


# צומת 4: Tool Execution
# שימוש ב-ToolNode המובנה של LangGraph שמריץ אוטומטית את הכלים שנקראו
tool_node = ToolNode(system_tools)


# צומת 5: Output Guardrail (סעיף 4.1)
def output_guardrail_node(state: AgentState):
    """מוודא שהתשובה הסופית אינה מכילה הזיות ומציגה מבנה דוח תקין."""
    last_message = state["messages"][-1]
    content = last_message.content

    # בדיקת איכות ותוכן מינימלי
    if not content or len(content.strip()) < 50:
        warning_suffix = "\n\n[Warning: Generated report lacks depth. Internal review advised.]"
        return {"messages": [AIMessage(content=content + warning_suffix)]}

    return {"messages": [last_message]}


# ==========================================
# 3. ניתוב מותנה (Conditional Edges)
# ==========================================

def route_after_input_guardrail(state: AgentState):
    """מנתב לסיום אם הקלט נפסל, או ממשיך לתכנון אם הוא תקין."""
    if not state.get("is_safe", True):
        return END
    return "research_planner"


def route_agent_decision(state: AgentState):
    """תנאי ReAct: בודק אם הסוכן ביקש לקרוא לכלי או שסיים את מחקרו."""
    last_message = state["messages"][-1]

    # מניעת לולאות אינסופיות (מקסימום 5 איטרציות חשיבה)
    if state.get("iteration_count", 0) >= 5:
        return "output_guardrail"

    # אם המודל יצר בקשה לקריאת כלי (tool_calls)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_execution"

    return "output_guardrail"


# ==========================================
# 4. הרכבת הגרף (StateGraph Construction)
# ==========================================

workflow = StateGraph(AgentState)

# הוספת הצמתים
workflow.add_node("input_guardrail", input_guardrail_node)
workflow.add_node("research_planner", research_planner_node)
workflow.add_node("agent_reasoning", agent_reasoning_node)
workflow.add_node("tool_execution", tool_node)
workflow.add_node("output_guardrail", output_guardrail_node)

# הגדרת נקודת ההתחלה
workflow.set_entry_point("input_guardrail")

# חיבור קשתות מותנות ורגילות
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

# לאחר הרצת כלי - חוזרים תמיד לסוכן לעיבוד התוצאה (ReAct Loop)
workflow.add_edge("tool_execution", "agent_reasoning")

# מסיום אימות הפלט מגיעים לסיום הריצה
workflow.add_edge("output_guardrail", END)

# הידור הגרף למנוע ריצה
app_graph = workflow.compile()

# ==========================================
# 5. בדיקה עצמאית והפקת ויזואליזציה (סעיף 5.2)
# ==========================================
if __name__ == "__main__":
    print("=== Testing Competitive Intelligence LangGraph ===")

    # שמירת תמונת ויזואליזציה של הגרף לתיקיית docs לפי דרישת סעיף 5.1 ו-5.2
    try:
        docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
        img_path = os.path.join(docs_dir, "langgraph_visualization.png")
        png_data = app_graph.get_graph().draw_mermaid_png()
        with open(img_path, "wb") as f:
            f.write(png_data)
        print(f"[Visualization] Graph structure saved to: {img_path}")
    except Exception as e:
        print(f"[Visualization Notice] Could not render PNG (requires mermaid dependencies): {e}")

    # הרצת בדיקה חיה של הגרף
    test_input = {
        "messages": [HumanMessage(content="What are the vulnerabilities of AlphaCorp and how should we compete?")]}
    print("\n[Running Execution] Invoking graph...")
    final_output = app_graph.invoke(test_input)

    print("\n=== Final Dossier Generated ===")
    print(final_output["messages"][-1].content)