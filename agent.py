"""
AutoStream Conversational AI Agent
Built with LangGraph + Groq (Free Tier)
Social-to-Lead Agentic Workflow for ServiceHive / Inflx Assignment
"""

import json
import os
import re
from typing import Annotated, TypedDict, Literal
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# ─────────────────────────────────────────────
# ✏️  PASTE YOUR GROQ API KEY BELOW
#     Get it free from: https://console.groq.com/keys
# ─────────────────────────────────────────────
GROQ_API_KEY = "PASTE_YOUR_GROQ_KEY_HERE"
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 1. KNOWLEDGE BASE LOADER (RAG)
# ─────────────────────────────────────────────


def load_knowledge_base() -> str:
    kb_path = Path(__file__).parent / "knowledge_base" / "autostream_kb.json"
    with open(kb_path, "r") as f:
        kb = json.load(f)

    sections = []
    sections.append(f"Company: {kb['company']} - {kb['tagline']}")
    sections.append("\n== PRICING PLANS ==")
    for plan in kb["plans"]:
        features = "\n  - ".join(plan["features"])
        sections.append(f"\n{plan['name']}: {plan['price']}\nFeatures:\n  - {features}")
    sections.append("\n\n== COMPANY POLICIES ==")
    for policy in kb["policies"]:
        sections.append(f"\n{policy['topic']}: {policy['detail']}")
    sections.append("\n\n== FAQ ==")
    for item in kb["faq"]:
        sections.append(f"\nQ: {item['question']}\nA: {item['answer']}")
    return "\n".join(sections)


KNOWLEDGE_BASE = load_knowledge_base()


# ─────────────────────────────────────────────
# 2. MOCK LEAD CAPTURE TOOL
# ─────────────────────────────────────────────


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    print(f"\n{'='*50}")
    print(f"✅ Lead captured successfully: {name}, {email}, {platform}")
    print(f"{'='*50}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"


# ─────────────────────────────────────────────
# 3. AGENT STATE
# ─────────────────────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: Literal["greeting", "product_inquiry", "high_intent", "unknown"]
    collecting_lead: bool
    lead_name: str | None
    lead_email: str | None
    lead_platform: str | None
    lead_captured: bool


# ─────────────────────────────────────────────
# 4. LLM SETUP (Groq - Free Tier)
#    Model: llama-3.3-70b-versatile (latest free model)
# ─────────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=GROQ_API_KEY,
)


# ─────────────────────────────────────────────
# 5. SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are an AI sales assistant for AutoStream, a SaaS platform providing automated video editing tools for content creators.

Your job is to:
1. Greet users warmly
2. Answer product and pricing questions accurately using ONLY the knowledge base below
3. Identify when users show HIGH INTENT (ready to sign up / try the product)
4. Collect lead information (name, email, platform) naturally in conversation

== KNOWLEDGE BASE ==
{KNOWLEDGE_BASE}

== INTENT DETECTION RULES ==
- GREETING: Simple hi/hello, small talk, no product interest yet
- PRODUCT_INQUIRY: Asking about features, pricing, policies, comparisons
- HIGH_INTENT: Phrases like "I want to sign up", "I want to try", "I'm ready", "sounds great let me get started", "how do I start", "I'll take the Pro plan", etc.

== LEAD COLLECTION RULES ==
- Only start collecting lead info when intent is HIGH_INTENT
- Collect: Full Name, Email Address, Creator Platform (YouTube, Instagram, TikTok, etc.)
- Ask for one piece at a time in a natural, friendly manner
- Do NOT ask for lead info if the user is just browsing/asking questions
- Once all 3 fields are collected, confirm and say the lead has been captured

== TONE ==
- Friendly, helpful, concise
- Never make up features or prices not in the knowledge base
- If unsure, say "I don't have that information, but our team can help!"
"""


# ─────────────────────────────────────────────
# 6. INTENT CLASSIFIER NODE
# ─────────────────────────────────────────────


def classify_intent(state: AgentState) -> AgentState:
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg:
        return {**state, "intent": "unknown"}

    classification_prompt = f"""Classify the following user message into exactly one of these intents:
- greeting
- product_inquiry
- high_intent

User message: "{last_user_msg}"

Reply with ONLY one word: greeting, product_inquiry, or high_intent"""

    result = llm.invoke([HumanMessage(content=classification_prompt)])
    raw = result.content.strip().lower()

    if "high_intent" in raw or "high intent" in raw:
        intent = "high_intent"
    elif "product" in raw or "inquiry" in raw:
        intent = "product_inquiry"
    elif "greeting" in raw:
        intent = "greeting"
    else:
        intent = "product_inquiry"

    return {**state, "intent": intent}


# ─────────────────────────────────────────────
# 7. EXTRACT LEAD INFO NODE
# ─────────────────────────────────────────────


def extract_lead_info(state: AgentState) -> AgentState:
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    updated = dict(state)

    # Extract email via regex
    if not updated.get("lead_email"):
        email_match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", last_user_msg)
        if email_match:
            updated["lead_email"] = email_match.group(0)

    # Extract platform via keyword matching
    if not updated.get("lead_platform"):
        platforms = [
            "youtube",
            "instagram",
            "tiktok",
            "twitter",
            "facebook",
            "linkedin",
            "snapchat",
        ]
        for p in platforms:
            if p in last_user_msg.lower():
                updated["lead_platform"] = p.capitalize()
                break

    # Extract name using LLM
    if not updated.get("lead_name"):
        name_prompt = f"""The user sent this message: "{last_user_msg}"

Does this message contain a person's full name? If yes, reply with ONLY the name.
If no name is present, reply with exactly: NONE"""
        name_result = llm.invoke([HumanMessage(content=name_prompt)])
        name_raw = name_result.content.strip()
        if (
            name_raw.upper() != "NONE"
            and len(name_raw.split()) >= 1
            and "@" not in name_raw
        ):
            updated["lead_name"] = name_raw

    return updated


# ─────────────────────────────────────────────
# 8. RESPONSE GENERATOR NODE
# ─────────────────────────────────────────────


def generate_response(state: AgentState) -> AgentState:
    updated = dict(state)

    lead_context = ""
    if state.get("collecting_lead") or state.get("intent") == "high_intent":
        collected = []
        missing = []
        if state.get("lead_name"):
            collected.append(f"Name: {state['lead_name']}")
        else:
            missing.append("Full Name")
        if state.get("lead_email"):
            collected.append(f"Email: {state['lead_email']}")
        else:
            missing.append("Email Address")
        if state.get("lead_platform"):
            collected.append(f"Platform: {state['lead_platform']}")
        else:
            missing.append("Creator Platform")

        lead_context = f"""
[LEAD COLLECTION STATUS]
Collected: {', '.join(collected) if collected else 'Nothing yet'}
Still need: {', '.join(missing) if missing else 'All collected!'}
"""

    # Fire tool only when all 3 fields are collected
    all_collected = (
        state.get("lead_name")
        and state.get("lead_email")
        and state.get("lead_platform")
        and not state.get("lead_captured")
    )

    if all_collected:
        mock_lead_capture(
            state["lead_name"], state["lead_email"], state["lead_platform"]
        )
        updated["lead_captured"] = True
        confirmation_msg = (
            f"🎉 You're all set, {state['lead_name']}! "
            f"We've captured your details and our team will reach out to your {state['lead_platform']} "
            f"account at {state['lead_email']}. Welcome to AutoStream! Your free Pro trial awaits."
        )
        updated["messages"] = state["messages"] + [AIMessage(content=confirmation_msg)]
        return updated

    messages_for_llm = [SystemMessage(content=SYSTEM_PROMPT)]
    if lead_context:
        messages_for_llm.append(SystemMessage(content=lead_context))
    messages_for_llm += state["messages"]

    intent_hint = f"[Current detected intent: {state.get('intent', 'unknown')}]"
    if state.get("intent") == "high_intent" and not state.get("collecting_lead"):
        intent_hint += (
            " The user is ready to sign up. Start collecting their lead info naturally."
        )
        updated["collecting_lead"] = True
    messages_for_llm.append(SystemMessage(content=intent_hint))

    response = llm.invoke(messages_for_llm)
    updated["messages"] = state["messages"] + [AIMessage(content=response.content)]
    return updated


# ─────────────────────────────────────────────
# 9. ROUTING LOGIC
# ─────────────────────────────────────────────


def route_after_intent(state: AgentState) -> str:
    if state.get("collecting_lead") or state.get("intent") == "high_intent":
        return "extract_lead"
    return "respond"


# ─────────────────────────────────────────────
# 10. BUILD LANGGRAPH
# ─────────────────────────────────────────────


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("extract_lead", extract_lead_info)
    graph.add_node("respond", generate_response)
    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {"extract_lead": "extract_lead", "respond": "respond"},
    )
    graph.add_edge("extract_lead", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


# ─────────────────────────────────────────────
# 11. MAIN CHAT LOOP
# ─────────────────────────────────────────────


def run_agent():
    app = build_graph()

    print("\n" + "=" * 60)
    print("🎬  AutoStream AI Assistant  |  Powered by Inflx / ServiceHive")
    print("=" * 60)
    print("Type 'quit' or 'exit' to end the conversation.\n")

    state: AgentState = {
        "messages": [],
        "intent": "unknown",
        "collecting_lead": False,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
    }

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print(
                "\nAssistant: Thanks for chatting with AutoStream! Have a great day 🎬"
            )
            break

        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]
        state = app.invoke(state)

        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                print(f"\nAssistant: {msg.content}\n")
                break


if __name__ == "__main__":
    run_agent()
