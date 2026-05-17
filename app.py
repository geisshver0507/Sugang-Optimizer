import streamlit as st
import google.generativeai as genai
from database import CS_COURSES
import json
import os

# 1. Initialize the native Google Gemini client
# Replace this with the real API key you got from Google AI Studio
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

st.title("🤖 Yonsei CS Course Consultant")
st.caption("Let's figure out your perfect schedule together before setting your mileage!")

# 2. Define our system rules for the AI
SYSTEM_INSTRUCTION = f"""
You are an interactive academic advisor for Yonsei University Computer Science students. 
Your goal is to collaboratively build a target course list for the upcoming semester.

CRITICAL RULES:
1. You have access to this EXACT course database: {json.dumps(CS_COURSES)}. Never suggest a course outside of this list.
2. Always be interactive. If the user doesn't have a list, ask about their workload tolerance or interest keywords. If they do have a list, evaluate it.
3. Every time you mention or suggest a course, provide a 1-sentence explanation using the 'review' or 'workload' data from the database so the user feels convinced.
4. Keep your responses conversational, natural, and student-friendly.
"""

# 3. Initialize chat history in Streamlit's memory if it doesn't exist
if "chat_session" not in st.session_state:
    # Set up the model configuration with our custom instructions
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_INSTRUCTION
    )
    # Start a live conversational chat session
    st.session_state.chat_session = model.start_chat(history=[])
    st.session_state.display_messages = []

# 4. Display past chat messages on the screen
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 5. Handle new user input
if user_input := st.chat_input("Ask about courses, list your current picks, or describe your ideal workload..."):
    
    # Display user's message in the UI
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.display_messages.append({"role": "user", "content": user_input})
    
    # Send the message to the live Gemini session and display the streaming response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        response = st.session_state.chat_session.send_message(user_input)
        ai_response = response.text
        
        response_placeholder.write(ai_response)
        
    # Save the AI response to display memory
    st.session_state.display_messages.append({"role": "assistant", "content": ai_response})