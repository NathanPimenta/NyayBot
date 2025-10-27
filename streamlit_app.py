import streamlit as st
import requests
from typing import Dict

# Configuration
API_BASE_URL = "http://localhost:8000"
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi"
}

# Streamlit UI Setup
st.set_page_config(
    page_title="NyayaBot - Legal QA Assistant",
    page_icon="⚖️",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stTextInput > div > div > input {
        padding: 12px !important;
    }
    .stButton > button {
        width: 100%;
        padding: 10px !important;
        background-color: #4CAF50 !important;
        color: white !important;
    }
    .stAlert {
        padding: 20px !important;
    }
    .source-card {
        padding: 15px;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.title("⚖️ NyayaBot - Legal QA Assistant")
st.markdown("Ask legal questions in English, Hindi, or Marathi and get citizen-friendly answers based on Indian legal documents.")

# Main Form
with st.form(key="qa_form"):
    # Question Input
    question = st.text_area(
        "Your Legal Question:",
        placeholder="Type your question here...",
        height=150
    )
    
    # Language Selection
    selected_lang = st.selectbox(
        "Language:",
        options=list(SUPPORTED_LANGUAGES.keys()),
        format_func=lambda x: SUPPORTED_LANGUAGES[x]
    )
    
    # Additional Options
    include_sources = st.checkbox("Include source documents", value=True)
    
    # Submit Button
    submitted = st.form_submit_button("Get Answer")

# Handle Form Submission
if submitted and question:
    with st.spinner("Analyzing your question..."):
        try:
            # Prepare API request
            payload = {
                "query": question,
                "language": selected_lang,
                "include_sources": include_sources
            }
            
            # Make API call
            response = requests.post(
                f"{API_BASE_URL}/ask",
                json=payload
            )
            
            if response.status_code == 200:
                result: Dict = response.json()
                
                # Display Answer
                st.subheader("Answer:")
                st.markdown(f"> {result['answer']}")
                
                # Display Sources if available
                if include_sources and result.get("sources"):
                    st.subheader("Relevant Legal Sources:")
                    for idx, source in enumerate(result["sources"], 1):
                        with st.expander(f"Source #{idx}: {source['source']}"):
                            st.markdown(f"**Page:** {source.get('page', 'N/A')}")
                            st.markdown(f"**Relevance Score:** {source['relevance_score']:.2f}")
                            st.markdown(f"**Excerpt:**\n{source['text']}")
                
                # Show original query details
                st.caption(f"Detected language: {SUPPORTED_LANGUAGES.get(result['language'], 'English')}")
                
            else:
                st.error(f"API Error: {response.text}")
                
        except requests.ConnectionError:
            st.error("Could not connect to the API. Please make sure the backend server is running.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

elif submitted and not question:
    st.warning("Please enter a question before submitting.")

# Sidebar Information
with st.sidebar:
    st.header("About NyayaBot")
    st.markdown("""
    NyayaBot is an AI-powered legal assistant that helps citizens:
    - Understand complex legal concepts
    - Find relevant legal provisions
    - Get answers in simple language
    """)
    
    st.markdown("### Supported Languages")
    for code, lang in SUPPORTED_LANGUAGES.items():
        st.markdown(f"- {lang} ({code})")
    
    st.markdown("### How It Works")
    st.markdown("""
    1. Enter your legal question
    2. Select your preferred language
    3. Get AI-powered answer with sources
    """)
    
    st.markdown("### API Status")
    try:
        health_resp = requests.get(f"{API_BASE_URL}/health")
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            st.success(f"API Status: {health_data['status'].capitalize()}")
            st.caption("Components:")
            for comp, status in health_data["components"].items():
                st.markdown(f"- {comp.capitalize()}: {status}")
        else:
            st.error("API Unavailable")
    except:
        st.error("API Unavailable")

# Footer
st.markdown("---")
st.caption("NyayaBot v1.0 | Access to justice through AI")