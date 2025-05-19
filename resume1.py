import streamlit as st
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Load API Key for Groq
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM (LLaMA3-8b)
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

# Extract text from uploaded PDF
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Build prompt and get LLM response
def get_groq_response(resume_text, job_desc):
    input_prompt = f"""
Hey, act like a skilled and experienced ATS (Applicant Tracking System)
with a deep understanding of software engineering, data science, data analytics, and big data roles.
Your task is to evaluate the following resume against the provided job description.

Be highly accurate and consider the competitive job market.

Resume:
{resume_text}

Job Description:
{job_desc}

Respond in this JSON format:
{{
  "JD Match": "XX%", 
  "MissingKeywords": [list of missing keywords], 
  "Profile Summary": "summary text here"
}}
"""
    response = llm.invoke(input_prompt)
    return response.content

# Streamlit App UI
st.title("Smart ATS")
st.text("Improve Your Resume for ATS Systems")

jd = st.text_area("Paste the Job Description")
uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload a PDF")

submit = st.button("Submit")

if submit:
    if uploaded_file is not None and jd:
        resume_text = input_pdf_text(uploaded_file)
        result = get_groq_response(resume_text, jd)
        st.subheader("Result")
        st.code(result, language="json")
    else:
        st.warning("Please upload a resume and provide a job description.")
