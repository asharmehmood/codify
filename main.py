
# For production
# __import__("pysqlite3")
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import streamlit as st
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory,ConversationBufferWindowMemory
from langchain.schema.runnable import RunnableConfig
from langsmith import Client
from streamlit_feedback import streamlit_feedback

from codify import code_generator

st.set_page_config(
    page_title="Codify - Your Coding Partner",
    page_icon="/*/",
)

# Set LangSmith environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Add the toggle for LangSmith API key source
use_secret_key = st.sidebar.toggle(label="Use Demo LangSmith API key", value=True)

# Conditionally set the project name based on the toggle
if use_secret_key:
    os.environ["LANGCHAIN_PROJECT"] = "Streamlit Demo"
else:
    project_name = st.sidebar.text_input(
        "Name your LangSmith Project:", value="Streamlit Demo"
    )
    os.environ["LANGCHAIN_PROJECT"] = project_name

# Conditionally get the API key based on the toggle
if use_secret_key:
    langchain_api_key = st.secrets["langchain"]["langsmith_api_key"] 
else:
    langchain_api_key = st.sidebar.text_input(
        "👇 Add your LangSmith Key",
        value="",
        placeholder="Your_LangSmith_Key_Here",
        label_visibility="collapsed",
    )
    
if langchain_api_key is not None:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

if "last_run" not in st.session_state:
    st.session_state["last_run"] = "hello"

langchain_endpoint = "https://api.smith.langchain.com"

col1, col2, col3 = st.columns([0.6, 3, 1])


st.markdown("<h1 style='text-align: center; color: teal;'>Codify - Your Coding Partner</h1>", unsafe_allow_html=True)

st.markdown("___")

# Check if the LangSmith API key is provided
if not langchain_api_key or langchain_api_key.strip() == "Your_LangSmith_Key_Here":
    st.info("⚠️ Add your [LangSmith API key](https://python.langchain.com/docs/guides/langsmith/walkthrough) to continue, or switch to the Demo key")
else:
    client = Client(api_url=langchain_endpoint, api_key=langchain_api_key)

# Initialize State
if "trace_link" not in st.session_state:
    st.session_state.trace_link = None
if "run_id" not in st.session_state:
    st.session_state.run_id = None

llm_type = st.sidebar.radio("Choose your LLM:",("mistral","gemini"),index=1)

memory = ConversationBufferWindowMemory(chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),k=5, memory_key='chat_history',return_messages=True)

cod = code_generator()
chain = cod.code_generator_chain(memory,llm_type)

if st.sidebar.button("Clear message history"):
    print("Clearing message history")
    memory.clear()
    st.session_state.trace_link = None
    st.session_state.run_id = None

# For older version of streamlit
def _get_openai_type(msg):
    if msg.type == "human":
        return "user"
    if msg.type == "ai":
        return "assistant"
    if msg.type == "chat":
        return msg.role
    return msg.type

# Initialize the modified list with the first two values
modified_list = st.session_state.langchain_messages[:2]

# Iterate over the remaining elements, skipping every second value
for i in range(4, len(st.session_state.langchain_messages), 4):
    modified_list.extend(st.session_state.langchain_messages[i:i+2])

for msg in modified_list:
    streamlit_type = _get_openai_type(msg)
    avatar = "🤖" if streamlit_type == "assistant" else None
    with st.chat_message(streamlit_type, avatar=avatar):
        st.markdown(msg.content)

run_collector = RunCollectorCallbackHandler()
runnable_config = RunnableConfig(
    callbacks=[run_collector],
    tags=["Streamlit Chat"],
)
if st.session_state.trace_link:
    st.sidebar.markdown(
        f'<a href="{st.session_state.trace_link}" target="_blank"><button>Debug Last Call: 🛠️</button></a>',
        unsafe_allow_html=True,
    )

def _reset_feedback():
    st.session_state.feedback_update = None
    st.session_state.feedback = None


MAX_CHAR_LIMIT = 500

if prompt := st.chat_input(placeholder="Tell me about your coding problems!"):

    if len(prompt) > MAX_CHAR_LIMIT:
        st.warning(f"⚠️ Your input is too long! Please limit your input to {MAX_CHAR_LIMIT} characters.")
        prompt = None  # Reset the prompt so it doesn't get processed further
    else:
        st.chat_message("user").write(prompt)
        _reset_feedback()
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""

            input_structure = {
                "user_query": prompt,
                "chat_history": [
                    (msg.type, msg.content)
                    for msg in st.session_state.langchain_messages
                ],
            }

            for chunk in chain.stream(input_structure, config=runnable_config):
                full_response += chunk["text"]
                message_placeholder.markdown(full_response + "▌")
            memory.save_context({"input": prompt}, {"output": full_response})

            message_placeholder.markdown(full_response)

            # The run collector will store all the runs in order. We'll just take the root and then
            # reset the list for next interaction.
            run = run_collector.traced_runs[0]
            run_collector.traced_runs = []
            st.session_state.run_id = run.id
            wait_for_all_tracers()
            url = client.share_run(run.id)
            # Or if you just want to use this internally
            # without sharing
            # url = client.read_run(run.id).url
            st.session_state.trace_link = url


has_chat_messages = len(st.session_state.get("langchain_messages", [])) > 0

# Only show the feedback toggle if there are chat messages
if has_chat_messages:
    feedback_option = (
        "faces" if st.toggle(label="`Thumbs` ⇄ `Faces`", value=False) else "thumbs"
    )
else:
    pass

if st.session_state.get("run_id"):
    feedback = streamlit_feedback(
        feedback_type=feedback_option,  # Use the selected feedback option
        optional_text_label="[Optional] Please provide an explanation",  # Adding a label for optional text input
        key=f"feedback_{st.session_state.run_id}",
    )

    # Define score mappings for both "thumbs" and "faces" feedback systems
    score_mappings = {
        "thumbs": {"👍": 1, "👎": 0},
        "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
    }

    # Get the score mapping based on the selected feedback option
    scores = score_mappings[feedback_option]

    if feedback:
        # Get the score from the selected feedback option's score mapping
        score = scores.get(feedback["score"])

        if score is not None:
            # Formulate feedback type string incorporating the feedback option and score value
            feedback_type_str = f"{feedback_option} {feedback['score']}"

            # Record the feedback with the formulated feedback type string and optional comment
            feedback_record = client.create_feedback(
                st.session_state.run_id,
                feedback_type_str,  # Updated feedback type
                score=score,
                comment=feedback.get("text"),
            )
            st.session_state.feedback = {
                "feedback_id": str(feedback_record.id),
                "score": score,
            }
        else:
            st.warning("Invalid feedback score.")