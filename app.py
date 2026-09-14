import streamlit as st
from krag_retrival import krag_retrieval_pipeline
from run_guardrails import get_rails_instance

# Page configuration
st.set_page_config(
    page_title="Tamil Nadu Government Schemes - Knowledge RAG Chatbot",
    page_icon="🌾",
    layout="wide",
)

# Cache the NeMo Guardrails instance across Streamlit reruns
@st.cache_resource(show_spinner="Loading NeMo Guardrails...")
def load_guardrails():
    return get_rails_instance()

rails = load_guardrails()

# Custom styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .guardrail-blocked {
        padding: 0.8rem;
        border-radius: 8px;
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        color: #e65100;
        margin-bottom: 0.8rem;
    }
    .guardrail-passed {
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        background-color: #e8f5e9;
        color: #2e7d32;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.title("🌾 About the Chatbot")
    st.markdown(
        """
        This chatbot uses a **Knowledge-Graph RAG (K-RAG)** architecture backed by **Neo4j**, **OpenAI**, and **NeMo Guardrails**:
        
        1. **🛡️ NeMo Guardrails**: Moderates user input, filters off-topic questions, and prevents jailbreaks.
        2. **Step 1**: Extracts entities & relationships from your question.
        3. **Step 2**: Matches nodes & graph patterns in Neo4j.
        4. **Step 3**: Generates a dynamic Cypher query.
        5. **Step 4**: Synthesizes the final answer using retrieved graph context.
        """
    )

    st.markdown("---")
    use_guardrails = st.checkbox("🛡️ Enable NeMo Guardrails", value=True)
    show_details = st.checkbox("🔍 Show Retrieval & Graph Details", value=True)

    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "Explain schema which has helps watering the crops?",
        "What schemes provide loans for sericulture in irrigated area?",
        "What is the stock price of Apple?",  # Tests off-topic guardrail
        "Tell me about training provided to farmers",
    ]

    for q in sample_questions:
        if st.button(q, key=f"btn_{q}"):
            st.session_state["pending_prompt"] = q

    st.markdown("---")
    st.caption("📁 **Guardrail Config Files:**")
    st.code("guardrails_config/config.yml\nguardrails_config/rails.co", language="text")

    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# Header
st.markdown('<div class="main-title">🌾 TN Schemes Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Ask any question about Tamil Nadu Government Schemes. Responses are grounded in a Neo4j Knowledge Graph and protected by NVIDIA NeMo Guardrails.</div>',
    unsafe_allow_html=True,
)

# Initialize chat session history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your Tamil Nadu Government Schemes assistant. Ask me anything about available agricultural subsidies, loans, irrigation projects, or farmer welfare schemes!",
            "details": None,
            "guardrail_status": "passed",
        }
    ]

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("guardrail_status") == "blocked":
            st.markdown(
                '<div class="guardrail-blocked">🛡️ <b>NeMo Guardrails Intercepted:</b> Off-topic or unsafe query prevented from reaching the database.</div>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])
        if msg.get("details") and show_details:
            with st.expander("🔍 View Knowledge Graph Retrieval Steps"):
                d = msg["details"]
                st.markdown("**Step 1: Extracted Entities & Relations**")
                st.json(d.get("step_1_extracted", {}))

                st.markdown("**Step 2: Matched Neo4j Nodes**")
                matched_nodes = d.get("step_2_matched", {}).get("matched_nodes", [])
                if matched_nodes:
                    st.table([{"ID": n.get("id"), "Labels": ", ".join(n.get("labels", []))} for n in matched_nodes[:8]])
                else:
                    st.info("No direct nodes matched in Step 2.")

                st.markdown("**Step 3: Generated Cypher Query**")
                st.code(d.get("step_3_cypher", ""), language="cypher")

                st.markdown(f"**Step 4: Neo4j Records Retrieved ({d.get('step_4_execution', {}).get('results_count', 0)})**")
                raw = d.get("step_4_execution", {}).get("raw_results", [])
                if raw:
                    st.json(raw[:5])

# Check for pending prompt from sample buttons or chat input
user_query = st.chat_input("Ask a question about government schemes...")
if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
    user_query = st.session_state.pop("pending_prompt")

if user_query:
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": user_query, "details": None})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Process query with NeMo Guardrails first (if enabled)
    with st.chat_message("assistant"):
        is_blocked = False
        guardrail_reply = ""

        if use_guardrails:
            with st.status("🛡️ Checking input with NeMo Guardrails...", expanded=False) as g_status:
                rail_res = rails.generate(messages=[{"role": "user", "content": user_query}])
                content = rail_res.get("content", "").strip() if isinstance(rail_res, dict) else str(rail_res).strip()

                refusal_phrases = [
                    "specialize only in tamil nadu",
                    "cannot answer unrelated",
                    "cannot comply",
                    "unrelated questions",
                    "can't provide information about stock prices",
                    "cannot provide information about stock prices",
                    "hello! i am your ai assistant",
                ]
                
                if any(phrase in content.lower() for phrase in refusal_phrases):
                    is_blocked = True
                    guardrail_reply = content
                    g_status.update(label="🛡️ NeMo Guardrails: Intercepted (Off-Topic / Policy Rail)", state="error", expanded=False)
                else:
                    g_status.update(label="🛡️ NeMo Guardrails: Input Approved (In-Domain)", state="complete", expanded=False)

        # If blocked by Guardrails, display refusal directly without querying Neo4j
        if is_blocked:
            st.markdown(
                '<div class="guardrail-blocked">🛡️ <b>NeMo Guardrails Intercepted:</b> This query is off-topic or violates safety guidelines. Neo4j Knowledge Graph search skipped.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(guardrail_reply)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": guardrail_reply,
                    "details": None,
                    "guardrail_status": "blocked",
                }
            )
        else:
            # 3. If approved, run the 4-step K-RAG pipeline
            with st.status("Searching Knowledge Graph...", expanded=show_details) as status:
                st.write("Step 1: Extracting entities & relationships from query...")
                result = krag_retrieval_pipeline(user_query, verbose=False)

                st.write(f"Step 2: Matched {len(result['step_2_matched'].get('matched_nodes', []))} nodes in Neo4j.")
                st.write("Step 3: Cypher query generated by LLM.")
                st.code(result.get("step_3_cypher", ""), language="cypher")

                st.write(f"Step 4: Retrieved {result['step_4_execution'].get('results_count', 0)} records. Synthesizing answer...")
                status.update(label="Response generated from Neo4j Knowledge Graph!", state="complete", expanded=False)

            final_answer = result["step_4_execution"]["final_answer"]
            st.markdown(final_answer)

            # Save to chat history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                    "details": result,
                    "guardrail_status": "passed",
                }
            )
