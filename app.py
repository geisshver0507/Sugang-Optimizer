import streamlit as st
from groq import Groq
from database import CS_COURSES
import json

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

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

# 3. Initialize chat history in Streamlit's memory
if "messages" not in st.session_state:
    # Groq is stateless — we maintain history ourselves as a list of dicts
    st.session_state.messages = []          # sent to Groq API (includes system msg)
    st.session_state.display_messages = []  # shown in the UI

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
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Send full conversation history to Groq (stateless API requires this)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # or "mixtral-8x7b-32768", "gemma2-9b-it"
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                *st.session_state.messages      # full history injected every call
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        ai_response = response.choices[0].message.content
        response_placeholder.write(ai_response)

    # Save the AI response to both histories
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.session_state.display_messages.append({"role": "assistant", "content": ai_response})