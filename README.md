# AutoStream AI Agent
### Social-to-Lead Agentic Workflow 

A conversational AI agent built for **AutoStream**, a SaaS platform for automated video editing. The agent handles product queries, identifies high-intent users, and captures qualified leads — end-to-end.

---


## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/mahivbhosale/autostream-agent.git
cd autostream-agent
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Get Your Free Groq API Key
- Go to: **https://console.groq.com/keys**
- Sign in with Google → Click **"Create API Key"** → Copy it
- Open `agent.py` and paste your key on **line 27**:
```python
GROQ_API_KEY = "your_groq_key_here"
```

### 5. Set Up the Knowledge Base Folder
```bash
# Windows
mkdir knowledge_base
copy autostream_kb.json knowledge_base\autostream_kb.json

# macOS/Linux
mkdir knowledge_base
cp autostream_kb.json knowledge_base/autostream_kb.json
```

### 6. Run the Agent
```bash
python agent.py
```

---

## 💬 Example Conversation

```
You: Hi there!
Assistant: Hey! Welcome to AutoStream. How can I help you today?

You: What are your pricing plans?
Assistant: We have two plans:
  - Basic Plan ($29/month): 10 videos/month, 720p, email support
  - Pro Plan ($79/month): Unlimited videos, 4K resolution, AI captions, 24/7 support

You: What is your refund policy?
Assistant: No refunds after 7 days. Requests within 7 days are processed in 3-5 business days.

You: That sounds great, I want to sign up for the Pro plan for my YouTube channel
Assistant: The Pro plan is a great choice! Before I get you set up, can you share your full name?

You: Riya Sharma
Assistant: Nice to meet you, Riya! Could you share your email address?

You: riya@example.com

==================================================
✅ Lead captured successfully: Riya Sharma, riya@example.com, Youtube
==================================================
---

## 🏗️ Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because it offers **explicit, inspectable state machines** — critical for a lead-capture flow where premature tool execution is a hard failure. With LangGraph, every conversation turn passes through a typed `AgentState` dictionary, making it trivial to track partial lead data (name/email/platform) across multiple turns without losing context.

### How State is Managed

The `AgentState` TypedDict holds:
- **`messages`** – Full conversation history (LangGraph's `add_messages` reducer appends, never overwrites)
- **`intent`** – Updated each turn by the `classify_intent` node using a focused LLM prompt
- **`lead_*` fields** – Incrementally populated across turns using regex (email), keyword matching (platform), and LLM extraction (name)
- **`lead_captured`** – Boolean gate preventing duplicate tool calls

The graph flows: `classify_intent → (conditional) → extract_lead → respond`, or `classify_intent → respond` for non-lead turns. This ensures the `mock_lead_capture()` tool is only triggered once all three fields are present — never prematurely.

State persists for the full session (5–6+ turns) entirely in-process, making it production-ready for further extension with Redis or a database-backed checkpointer.

---

## 📱 WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, here is the recommended approach:

### Architecture

```
WhatsApp User
     │
     ▼
WhatsApp Business API (Meta)
     │  Webhook POST (JSON)
     ▼
FastAPI / Flask Server
     │
     │  1. Extract sender ID + message text
     │  2. Load session state from Redis (by sender ID)
     │  3. Run LangGraph agent.invoke(state)
     │  4. Save updated state back to Redis
     │  5. POST reply to WhatsApp API
```

### Step-by-Step

1. **Register a WhatsApp Business App** on Meta for Developers. Get a `PHONE_NUMBER_ID` and `WHATSAPP_TOKEN`.

2. **Set up a Webhook endpoint** (e.g., `/webhook`) using FastAPI:
```python
@app.post("/webhook")
async def receive_message(payload: dict):
    sender_id = payload["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
    user_text = payload["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]

    # Load state from Redis
    state = redis_client.get(f"session:{sender_id}") or default_state()

    # Run agent
    state["messages"].append(HumanMessage(content=user_text))
    new_state = langgraph_app.invoke(state)

    # Save state
    redis_client.set(f"session:{sender_id}", new_state, ex=3600)

    # Send reply
    reply = get_last_ai_message(new_state)
    send_whatsapp_message(sender_id, reply)
```

3. **Verify webhook** with Meta's GET verification handshake using a `VERIFY_TOKEN`.

4. **Handle session expiry**: Set a TTL (e.g., 1 hour) on Redis keys to auto-clear stale sessions.

5. **Deploy** on Railway, Render, or AWS Lambda with a public HTTPS URL for the webhook.

### Key Libraries
- `fastapi` + `uvicorn` for the server
- `redis` for per-user state persistence
- `httpx` for calling the WhatsApp API
- `python-dotenv` for secrets management

---

## 📁 Project Structure

```
autostream-agent/
├── agent.py                    # Main agent logic (LangGraph)
├── knowledge_base/
│   └── autostream_kb.json      # RAG knowledge base
├── autostream_kb.json          # Source KB file
├── requirements.txt
└── README.md
```

---

## 🎯 Evaluation Checklist

| Criteria | Implementation |
|---|---|
| Intent Detection | 3-class LLM classifier per turn (greeting / product_inquiry / high_intent) |
| RAG | JSON knowledge base loaded and injected into system prompt |
| State Management | LangGraph TypedDict with add_messages reducer across 5–6 turns |
| Tool Calling | mock_lead_capture() triggered only after all 3 fields collected |
| Memory | Full message history retained in AgentState |
| Code Clarity | Modular nodes, typed state, inline comments throughout |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Framework | LangGraph (LangChain ecosystem) |
| LLM | Llama 3.3 70B via Groq (Free Tier) |
| RAG | Local JSON knowledge base injected via system prompt |
| State | LangGraph TypedDict with add_messages reducer |
| API | Groq API (free at console.groq.com) |
