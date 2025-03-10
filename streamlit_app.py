#!/usr/bin/env python3
"""
Streamlit web interface for the ANUS framework.
This provides a simple visual interface for interacting with the ANUS AI agent.
"""

import os
import streamlit as st
import datetime
from dotenv import load_dotenv
from anus.core.orchestrator import AgentOrchestrator

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="ANUS - Autonomous Networked Utility System",
    page_icon="🍑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🍑 ANUS</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='sub-header'>Autonomous Networked Utility System</h2>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")
config_path = st.sidebar.text_input("Config Path", value=".anus/config.yaml")
agent_mode = st.sidebar.selectbox("Agent Mode", ["single", "multi"])
verbose = st.sidebar.checkbox("Verbose Output", value=True)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Main content
st.subheader("Ask ANUS")
user_input = st.text_area("Enter your task or question:", height=100)

# Execute button
if st.button("Execute Task"):
    if user_input:
        with st.spinner("ANUS is processing your request..."):
            try:
                # Initialize the agent orchestrator
                orchestrator = AgentOrchestrator(config_path=config_path)
                
                # Execute the task
                result = orchestrator.execute_task(user_input, mode=agent_mode)
                
                # Add to history
                st.session_state.history.append({
                    "task": user_input,
                    "result": result,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Display result
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                st.subheader("Task Result")
                st.write(f"**Task:** {user_input}")
                st.write("**Answer:**")
                st.write(result)
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a task or question.")

# History section
if st.session_state.history:
    st.subheader("History")
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Task: {item['task'][:50]}... ({item['timestamp']})"):
            st.write(f"**Task:** {item['task']}")
            st.write("**Result:**")
            st.write(item['result'])

# Footer
st.markdown("<div class='footer'>ANUS - Autonomous Networked Utility System</div>", unsafe_allow_html=True) 