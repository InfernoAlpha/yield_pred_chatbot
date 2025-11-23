import streamlit as st
import requests
import json

api_url = "http://13.60.76.254:8000/"

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_thread_ids"]:
        st.session_state["chat_thread_ids"].append(thread_id)

def reset_chat():
    id = requests.get(api_url + "uuid").json()
    st.session_state['thread_id'] = id["uuid"]
    add_thread(st.session_state['thread_id'])
    st.session_state['history'] = []

def load_conv(thread_id):
    state = json.loads(requests.post(url=api_url + "get_state",json={"thread_id":thread_id}).json()["state"])
    return state

if "history" not in st.session_state:
    st.session_state['history'] = []

if 'chat_thread_ids' not in st.session_state:
    st.session_state['chat_thread_ids'] = requests.get(api_url + "get_all_threads").json()['threads']

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = st.session_state['chat_thread_ids'][-1] if len(st.session_state['chat_thread_ids']) > 0 else []

add_thread(st.session_state['thread_id'])

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

length = len(st.session_state['chat_thread_ids'][::-1])
for num,thread_id in enumerate(st.session_state['chat_thread_ids'][::-1]):
    if st.sidebar.button("conversation: "+str(length - num)):
        st.session_state['thread_id'] = thread_id
        messages = load_conv(thread_id)

        temp_messages = []

        for msg in messages:
            if msg["kwargs"]["type"] == "human":
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg["kwargs"]["content"]})

        st.session_state['history'] = temp_messages

for message in st.session_state['history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('chat with me')

if user_input:
    st.session_state['history'].append({'role':'user','content':user_input})
    with st.chat_message("user"):
        st.text(user_input)
    
    CONFIG = {"configurable":{'thread_id':st.session_state['thread_id']},"metadata":{"thread_id":st.session_state["thread_id"]},"run_name":"chat_turn"}

    with st.chat_message("assistant"):
        status_holder = {"box":None}
        def ai_stream():
            stream_chunk = requests.post(url= api_url + "prompt",json={"thread_id":st.session_state["thread_id"],"input":user_input,"mode":"stream"},stream=True)
            for line in stream_chunk.iter_lines():
                message_chunk = json.loads(line)
                if message_chunk[0]['id'][3] == 'ToolMessage':
                    tool_name = message_chunk[0]['kwargs']['name']
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if message_chunk[0]['kwargs']['type'] == 'AIMessageChunk':
                    yield message_chunk[0]['kwargs']['content']

        ai_message = st.write_stream(ai_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state['history'].append({'role':'assistant','content':ai_message})