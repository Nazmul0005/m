import streamlit as st
import requests
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="FixConnect Chatbot",
    page_icon="🔧",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    color: #000; /* Ensure text is visible in light backgrounds */
}

.user-message {
    background-color: #e6f3ff;
    border-left: 5px solid #2196F3;
    color: #000;  /* black text for readability */
}

.bot-message {
    background-color: #f0f2f6;
    border-left: 5px solid #4CAF50;
    color: #000;  /* black text for readability */
}

.timestamp {
    font-size: 0.8rem;
    color: #666;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Header
st.title("🔧 FixConnect Chatbot")
st.markdown("""
Welcome to FixConnect support! I can help you with:
- Information about our services
- Registration and account help
- Booking services
- Payment inquiries
- Technical support
""")

# Chat interface
def send_message(message):
    url = "http://localhost:8000/chat"  # FastAPI endpoint
    payload = {
        "question": message,
        "user_id": "streamlit_user"
    }
    try:
        response = requests.post(url, json=payload)
        return response.json()["answer"]
    except Exception as e:
        return f"Error: Unable to get response from server. Please try again. ({str(e)})"

# Chat input
user_message = st.chat_input("Type your message here...")

if user_message:
    # Add user message to chat history
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": user_message, "time": timestamp})
    
    # Get bot response
    bot_response = send_message(user_message)
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "assistant", "content": bot_response, "time": timestamp})

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div>🧑 You: {message["content"]}</div>
            <div class="timestamp">{message["time"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <div>🤖 FixConnect: {message["content"]}</div>
            <div class="timestamp">{message["time"]}</div>
        </div>
        """, unsafe_allow_html=True)

# Sidebar with additional information
with st.sidebar:
    st.header("About FixConnect")
    st.markdown("""
    FixConnect is your trusted platform for finding certified engineers for:
    - Electrical Services
    - HVAC Systems
    - Security Systems
    - Power Solutions
    - Plumbing Services
    """)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()