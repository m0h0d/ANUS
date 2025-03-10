#!/usr/bin/env python3
"""
Autonomous Streamlit web interface with tools for routine tasks.
This provides a functional visual interface for interacting with AI that can perform autonomous tasks.
"""

import os
import streamlit as st
import datetime
import openai
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import time
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please add it to your .env file.")

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
    .tool-container {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .tool-result {
        background-color: #f1f3f5;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #4ECDC4;
        margin-top: 5px;
        margin-bottom: 15px;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Define tools for autonomous tasks
class Tools:
    @staticmethod
    def search_web(query, num_results=5):
        """Search the web for information."""
        try:
            st.info(f"🔍 Searching the web for: {query}")
            
            # Simulate web search with a delay
            time.sleep(2)
            
            # This is a mock implementation - in a real app, you would use a search API
            results = [
                {"title": f"Result {i+1} for {query}", "snippet": f"This is a snippet of information about {query}...", "url": f"https://example.com/result{i+1}"}
                for i in range(num_results)
            ]
            
            # Format results
            formatted_results = "\n\n".join([f"**{r['title']}**\n{r['snippet']}\n[Link]({r['url']})" for r in results])
            return formatted_results
        except Exception as e:
            return f"Error searching the web: {str(e)}"
    
    @staticmethod
    def analyze_website(url):
        """Analyze a website and provide improvement suggestions."""
        try:
            st.info(f"🔍 Analyzing website: {url}")
            
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            # Parse URL to get domain
            domain = urlparse(url).netloc
            
            # Simulate website analysis with a delay
            time.sleep(3)
            
            # This is a mock implementation - in a real app, you would actually analyze the website
            analysis = {
                "domain": domain,
                "load_time": f"{round(2 + 3 * (hash(domain) % 10) / 10, 2)}s",
                "mobile_friendly": "Yes" if hash(domain) % 2 == 0 else "No",
                "seo_score": f"{50 + hash(domain) % 50}/100",
                "accessibility_score": f"{60 + hash(domain) % 40}/100",
                "issues": [
                    "Slow loading images" if hash(domain) % 3 == 0 else "Good image optimization",
                    "Missing meta descriptions" if hash(domain) % 5 == 0 else "Good meta descriptions",
                    "Poor mobile responsiveness" if hash(domain) % 2 == 1 else "Good mobile responsiveness",
                    "Broken links detected" if hash(domain) % 7 == 0 else "No broken links",
                    "Missing alt text on images" if hash(domain) % 4 == 0 else "Good alt text usage"
                ],
                "recommendations": [
                    "Optimize images" if hash(domain) % 3 == 0 else "Continue good image practices",
                    "Add meta descriptions" if hash(domain) % 5 == 0 else "Maintain good SEO practices",
                    "Improve mobile layout" if hash(domain) % 2 == 1 else "Continue good mobile design",
                    "Fix broken links" if hash(domain) % 7 == 0 else "Maintain link integrity",
                    "Add alt text to images" if hash(domain) % 4 == 0 else "Continue accessibility practices"
                ]
            }
            
            # Format analysis
            formatted_analysis = f"""
### Website Analysis for {domain}

**Performance Metrics:**
- Load Time: {analysis['load_time']}
- Mobile Friendly: {analysis['mobile_friendly']}
- SEO Score: {analysis['seo_score']}
- Accessibility Score: {analysis['accessibility_score']}

**Issues Detected:**
{"".join(['- ' + issue + '\n' for issue in analysis['issues']])}

**Recommendations:**
{"".join(['- ' + rec + '\n' for rec in analysis['recommendations']])}
            """
            
            return formatted_analysis
        except Exception as e:
            return f"Error analyzing website: {str(e)}"
    
    @staticmethod
    def generate_report(data, report_type="general"):
        """Generate a report based on provided data."""
        try:
            st.info(f"📊 Generating {report_type} report...")
            
            # Simulate report generation with a delay
            time.sleep(2)
            
            # Parse data if it's a string
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except:
                    pass
            
            # This is a mock implementation - in a real app, you would generate a real report
            if report_type == "website_improvement":
                # Create a sample dataframe for websites
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
                    websites = data
                else:
                    websites = ["example.com", "sample-site.org", "testwebsite.net", "demo-page.com", "mockup.io"]
                
                df = pd.DataFrame({
                    "Website": websites,
                    "Load Time (s)": [round(1 + 5 * (hash(site) % 10) / 10, 2) for site in websites],
                    "Mobile Score": [60 + hash(site) % 40 for site in websites],
                    "SEO Score": [50 + hash(site) % 50 for site in websites],
                    "Accessibility": [70 + hash(site) % 30 for site in websites],
                    "Priority": ["High" if hash(site) % 3 == 0 else "Medium" if hash(site) % 3 == 1 else "Low" for site in websites]
                })
                
                # Create a bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                df.set_index('Website')[['Mobile Score', 'SEO Score', 'Accessibility']].plot(kind='bar', ax=ax)
                plt.title('Website Performance Metrics')
                plt.ylabel('Score')
                plt.tight_layout()
                
                # Display the dataframe and chart in Streamlit
                st.write("### Website Improvement Report")
                st.dataframe(df)
                st.pyplot(fig)
                
                # Generate text report
                report = f"""
## Website Improvement Priority Report

This report analyzes {len(websites)} websites and provides recommendations for improvements.

### Summary of Findings:
- {len([p for p in df['Priority'] if p == 'High'])} websites require high priority attention
- {len([p for p in df['Priority'] if p == 'Medium'])} websites need medium priority improvements
- {len([p for p in df['Priority'] if p == 'Low'])} websites have low priority issues

### Recommendations:

{chr(10).join([f"**{row['Website']}** ({row['Priority']} Priority):\n- {'Optimize images and reduce server response time' if row['Load Time (s)'] > 3 else 'Good loading performance'}\n- {'Improve mobile responsiveness' if row['Mobile Score'] < 80 else 'Good mobile design'}\n- {'Enhance SEO elements' if row['SEO Score'] < 80 else 'Good SEO practices'}\n" for _, row in df.iterrows()])}

### Next Steps:
1. Address high priority websites first
2. Schedule improvements for medium priority sites
3. Monitor low priority websites for any changes
                """
                
                return report
            else:
                return "Report type not supported. Please specify a valid report type."
        except Exception as e:
            return f"Error generating report: {str(e)}"
    
    @staticmethod
    def create_content_plan(topic, num_items=5):
        """Create a content plan for a given topic."""
        try:
            st.info(f"📝 Creating content plan for: {topic}")
            
            # Simulate content plan creation with a delay
            time.sleep(2)
            
            # This is a mock implementation - in a real app, you would generate a real content plan
            current_month = datetime.datetime.now().strftime("%B %Y")
            next_month = (datetime.datetime.now() + datetime.timedelta(days=30)).strftime("%B %Y")
            
            plan = f"""
## Content Plan for "{topic}"

### Month 1 ({current_month}):

1. **Blog Post**: "Introduction to {topic}: What You Need to Know"
   - Format: Long-form article (1500+ words)
   - Keywords: {topic}, introduction, basics
   - Target audience: Beginners
   - Deadline: {(datetime.datetime.now() + datetime.timedelta(days=7)).strftime("%B %d, %Y")}

2. **Infographic**: "{topic} at a Glance"
   - Format: Visual infographic
   - Distribution: Social media, blog
   - Target audience: Visual learners, social media users
   - Deadline: {(datetime.datetime.now() + datetime.timedelta(days=14)).strftime("%B %d, %Y")}

3. **Video Tutorial**: "Getting Started with {topic}"
   - Format: 5-10 minute tutorial video
   - Platform: YouTube, embedded on blog
   - Target audience: Visual learners, beginners
   - Deadline: {(datetime.datetime.now() + datetime.timedelta(days=21)).strftime("%B %d, %Y")}

### Month 2 ({next_month}):

4. **Case Study**: "How Company X Succeeded with {topic}"
   - Format: Long-form article with data
   - Keywords: {topic} case study, success story
   - Target audience: Decision-makers, intermediate users
   - Deadline: {(datetime.datetime.now() + datetime.timedelta(days=35)).strftime("%B %d, %Y")}

5. **Expert Interview**: "Insights on {topic} from Industry Leaders"
   - Format: Q&A style article or podcast
   - Keywords: {topic} expert, industry insights
   - Target audience: All levels, industry professionals
   - Deadline: {(datetime.datetime.now() + datetime.timedelta(days=42)).strftime("%B %d, %Y")}

### Content Distribution Strategy:
- Publish blog content on company website
- Share all content on social media (LinkedIn, Twitter, Facebook)
- Send newsletter to subscribers featuring new content
- Repurpose content for different platforms (e.g., turn blog posts into videos)

### Success Metrics:
- Page views and time on page
- Social media engagement (shares, likes, comments)
- Lead generation and conversions
- Keyword rankings for target terms
            """
            
            return plan
        except Exception as e:
            return f"Error creating content plan: {str(e)}"

# Function to process user input with autonomous capabilities
def process_with_tools(user_input, model, temperature, max_tokens):
    # Define the system message with tool descriptions
    system_message = """
You are ANUS (Autonomous Networked Utility System), an autonomous AI assistant that can perform various tasks.
You have access to the following tools:

1. search_web(query, num_results=5): Search the web for information
2. analyze_website(url): Analyze a website and provide improvement suggestions
3. generate_report(data, report_type="general"): Generate a report based on provided data
4. create_content_plan(topic, num_items=5): Create a content plan for a given topic

When a user asks you to perform a task, determine which tool(s) to use and how to use them.
Format your response as follows:

THOUGHT: Your reasoning about what tools to use and how to approach the task
TOOL: tool_name(parameters)
ACTION: Description of what you're doing with the tool
RESULT: The result from the tool (this will be filled in by the system)
NEXT: Your next step, which could be using another tool or providing a final answer
FINAL ANSWER: Your complete response to the user's request

Always start with THOUGHT and end with FINAL ANSWER.
"""

    # Initial call to determine which tools to use
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Extract the initial response
    ai_message = response.choices[0].message.content
    
    # Process the response to execute tools
    processed_response = ""
    final_answer = ""
    
    # Track all steps for display
    execution_steps = []
    
    # Process the AI's response and execute tools
    sections = re.split(r'(THOUGHT:|TOOL:|ACTION:|RESULT:|NEXT:|FINAL ANSWER:)', ai_message)
    sections = [s.strip() for s in sections if s.strip()]
    
    i = 0
    while i < len(sections):
        section_type = sections[i]
        
        if i + 1 < len(sections):
            content = sections[i + 1]
            
            if section_type == "THOUGHT:":
                execution_steps.append({"type": "thought", "content": content})
            
            elif section_type == "TOOL:":
                # Extract tool name and parameters
                tool_match = re.match(r'(\w+)\((.*)\)', content)
                if tool_match:
                    tool_name = tool_match.group(1)
                    params_str = tool_match.group(2)
                    
                    # Parse parameters
                    params = []
                    kwargs = {}
                    
                    if params_str:
                        # Simple parameter parsing (this is a basic implementation)
                        param_parts = params_str.split(',')
                        for part in param_parts:
                            if '=' in part:
                                key, value = part.split('=', 1)
                                key = key.strip()
                                value = value.strip()
                                
                                # Convert value to appropriate type
                                if value.startswith('"') and value.endswith('"'):
                                    value = value[1:-1]
                                elif value.startswith("'") and value.endswith("'"):
                                    value = value[1:-1]
                                elif value.lower() == 'true':
                                    value = True
                                elif value.lower() == 'false':
                                    value = False
                                elif value.isdigit():
                                    value = int(value)
                                elif re.match(r'^-?\d+(\.\d+)?$', value):
                                    value = float(value)
                                
                                kwargs[key] = value
                            else:
                                part = part.strip()
                                if part.startswith('"') and part.endswith('"'):
                                    params.append(part[1:-1])
                                elif part.startswith("'") and part.endswith("'"):
                                    params.append(part[1:-1])
                                elif part.lower() == 'true':
                                    params.append(True)
                                elif part.lower() == 'false':
                                    params.append(False)
                                elif part.isdigit():
                                    params.append(int(part))
                                elif re.match(r'^-?\d+(\.\d+)?$', part):
                                    params.append(float(part))
                                else:
                                    params.append(part)
                    
                    # Execute the tool
                    tool_result = "Tool execution failed: Tool not found"
                    
                    if tool_name == "search_web" and hasattr(Tools, "search_web"):
                        if params:
                            tool_result = Tools.search_web(*params, **kwargs)
                        elif kwargs:
                            tool_result = Tools.search_web(**kwargs)
                    
                    elif tool_name == "analyze_website" and hasattr(Tools, "analyze_website"):
                        if params:
                            tool_result = Tools.analyze_website(*params, **kwargs)
                        elif kwargs:
                            tool_result = Tools.analyze_website(**kwargs)
                    
                    elif tool_name == "generate_report" and hasattr(Tools, "generate_report"):
                        if params:
                            tool_result = Tools.generate_report(*params, **kwargs)
                        elif kwargs:
                            tool_result = Tools.generate_report(**kwargs)
                    
                    elif tool_name == "create_content_plan" and hasattr(Tools, "create_content_plan"):
                        if params:
                            tool_result = Tools.create_content_plan(*params, **kwargs)
                        elif kwargs:
                            tool_result = Tools.create_content_plan(**kwargs)
                    
                    execution_steps.append({"type": "tool", "name": tool_name, "params": params, "kwargs": kwargs})
                    execution_steps.append({"type": "result", "content": tool_result})
                    
                    # Update the AI with the tool result
                    response = openai.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_input},
                            {"role": "assistant", "content": ai_message},
                            {"role": "user", "content": f"RESULT: {tool_result}\n\nContinue your reasoning and use additional tools if needed, or provide a final answer."}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    # Update AI message with the continuation
                    continuation = response.choices[0].message.content
                    ai_message += f"\nRESULT: {tool_result}\n\n{continuation}"
                    
                    # Extract new sections from the continuation
                    new_sections = re.split(r'(THOUGHT:|TOOL:|ACTION:|RESULT:|NEXT:|FINAL ANSWER:)', continuation)
                    new_sections = [s.strip() for s in new_sections if s.strip()]
                    
                    # Replace the current sections with the updated ones
                    sections = sections[:i+2] + new_sections
            
            elif section_type == "ACTION:":
                execution_steps.append({"type": "action", "content": content})
            
            elif section_type == "NEXT:":
                execution_steps.append({"type": "next", "content": content})
            
            elif section_type == "FINAL ANSWER:":
                final_answer = content
                execution_steps.append({"type": "final", "content": content})
                break
        
        i += 2
    
    return final_answer, execution_steps

# Header
st.markdown("<h1 class='main-header'>🍑 ANUS</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='sub-header'>Autonomous Networked Utility System</h2>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")
model = st.sidebar.selectbox(
    "Model", 
    ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    index=0
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.sidebar.slider("Max Tokens", 100, 4000, 1500, 100)
show_steps = st.sidebar.checkbox("Show Execution Steps", value=True)

# Tool showcase
st.sidebar.title("Available Tools")
st.sidebar.markdown("""
- 🔍 **Web Search**: Find information online
- 🌐 **Website Analysis**: Get improvement suggestions
- 📊 **Report Generation**: Create data reports
- 📝 **Content Planning**: Develop content strategies
""")

# Example tasks
st.sidebar.title("Example Tasks")
example_tasks = [
    "Analyze these websites and tell me which ones need improvement: example.com, mysite.org, testsite.net",
    "Create a content plan for digital marketing",
    "Generate a report on website performance for these sites: example.com, mysite.org, testsite.net",
    "Search for information about the latest web design trends and summarize them"
]

selected_example = st.sidebar.selectbox("Try an example:", [""] + example_tasks)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Main content
st.subheader("Ask ANUS to perform autonomous tasks")
user_input = st.text_area("Enter your task or question:", value=selected_example if selected_example else "", height=100)

# Execute button
if st.button("Execute Task"):
    if user_input:
        with st.spinner("ANUS is autonomously processing your request..."):
            try:
                # Process with tools
                final_answer, execution_steps = process_with_tools(user_input, model, temperature, max_tokens)
                
                # Add to history
                st.session_state.history.append({
                    "task": user_input,
                    "result": final_answer,
                    "steps": execution_steps,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Display result
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                st.subheader("Task Result")
                st.write(f"**Task:** {user_input}")
                
                # Show execution steps if enabled
                if show_steps:
                    st.write("**Execution Steps:**")
                    for step in execution_steps:
                        if step["type"] == "thought":
                            st.markdown(f"💭 **Thought:** {step['content']}")
                        elif step["type"] == "tool":
                            params_str = ", ".join([str(p) for p in step.get("params", [])])
                            kwargs_str = ", ".join([f"{k}={v}" for k, v in step.get("kwargs", {}).items()])
                            all_params = ", ".join(filter(None, [params_str, kwargs_str]))
                            st.markdown(f"🔧 **Tool:** {step['name']}({all_params})")
                        elif step["type"] == "action":
                            st.markdown(f"▶️ **Action:** {step['content']}")
                        elif step["type"] == "result":
                            st.markdown(f"🔍 **Tool Result:**")
                            st.markdown(f"<div class='tool-result'>{step['content']}</div>", unsafe_allow_html=True)
                        elif step["type"] == "next":
                            st.markdown(f"⏭️ **Next Step:** {step['content']}")
                
                st.write("**Final Answer:**")
                st.markdown(final_answer)
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
            
            if show_steps and "steps" in item:
                st.write("**Execution Steps:**")
                for step in item["steps"]:
                    if step["type"] == "thought":
                        st.markdown(f"💭 **Thought:** {step['content']}")
                    elif step["type"] == "tool":
                        params_str = ", ".join([str(p) for p in step.get("params", [])])
                        kwargs_str = ", ".join([f"{k}={v}" for k, v in step.get("kwargs", {}).items()])
                        all_params = ", ".join(filter(None, [params_str, kwargs_str]))
                        st.markdown(f"🔧 **Tool:** {step['name']}({all_params})")
                    elif step["type"] == "action":
                        st.markdown(f"▶️ **Action:** {step['content']}")
                    elif step["type"] == "result":
                        st.markdown(f"🔍 **Tool Result:**")
                        st.markdown(f"<div class='tool-result'>{step['content']}</div>", unsafe_allow_html=True)
                    elif step["type"] == "next":
                        st.markdown(f"⏭️ **Next Step:** {step['content']}")
            
            st.write("**Final Answer:**")
            st.markdown(item['result'])

# Footer
st.markdown("<div class='footer'>ANUS - Autonomous Networked Utility System</div>", unsafe_allow_html=True) 