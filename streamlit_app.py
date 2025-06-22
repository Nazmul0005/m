import streamlit as st
import requests
import json
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="MRPH Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.chat-container {
    border-radius: 10px;
    margin-bottom: 10px;
    padding: 15px;
}
.user-message {
    background-color: #e6f3ff;
    border-left: 5px solid #2196F3;
            color: #000;
}
.bot-message {
    background-color: #f0f2f6;
    border-left: 5px solid #4CAF50;
            color: #000;
}
.chat-header {
    font-weight: bold;
    margin-bottom: 5px;
}
.chat-timestamp {
    color: #666;
    font-size: 0.8em;
    margin-top: 5px;
}
.service-category {
    padding: 10px;
    border-radius: 5px;
    background-color: #f8f9fa;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar with categories from MRPH_CATEGORIES
with st.sidebar:
    st.title("🔧 MRPH Services")
    
    st.markdown("### Service Categories")
    st.markdown("""
    - ⚡ Electrical Services
    - 🌡️ HVAC Systems
    - 🚰 Plumbing Services
    - 🔒 Security Systems
    - ⚡ Power Solutions
    """)
    
    st.markdown("### Quick Links")
    st.markdown("""
    - 📝 Registration
    - 📅 Book a Service
    - 💳 Payments
    - 📱 Account Management
    - ❓ Help & Support
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("MRPH Support Assistant")
st.markdown("""
Welcome! I can help you with:
- Finding and booking services
- Account registration and management
- Scheduling appointments
- Payment and billing inquiries
- Technical support
""")

# Chat input
user_input = st.chat_input("Type your message here...")

# Handle user input
if user_input:
    # Add user message to chat history
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })
    
    # Get bot response
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={
                "question": user_input,
                "user_id": "streamlit_user"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            bot_response = response.json()["answer"]
        else:
            bot_response = "Sorry, I encountered an error. Please try again."
            
    except requests.exceptions.RequestException as e:
        bot_response = "Sorry, I'm having trouble connecting to the server. Please try again."
    
    # Add bot response to chat history
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_response,
        "timestamp": timestamp
    })

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-container user-message">
            <div class="chat-header">You</div>
            <div>{message["content"]}</div>
            <div class="chat-timestamp">{message["timestamp"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-container bot-message">
            <div class="chat-header">MRPH Assistant</div>
            <div>{message["content"]}</div>
            <div class="chat-timestamp">{message["timestamp"]}</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "MRPH - Find Trusted Engineers Fast. Connect with certified professionals for any technical issue.",
    help="MRPH is a mobile app that connects customers with certified engineers for technical services."
)