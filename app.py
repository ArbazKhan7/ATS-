import streamlit as st
from dotenv import load_dotenv
import os
import PyPDF2 as pdf
from langchain_groq import ChatGroq
from fpdf import FPDF

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# LLM setup (Groq + LLaMA3)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Extract text from PDF
def extract_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Save improved resume as PDF
def save_text_as_pdf(text, filename="Improved_Resume.pdf"):
    pdf_obj = FPDF()
    pdf_obj.add_page()
    pdf_obj.add_font("DejaVu", "", fname="DejaVuSans.ttf", uni=True)
    pdf_obj.set_font("DejaVu", size=12)

    for line in text.split("\n"):
        pdf_obj.multi_cell(0, 10, line)

    pdf_path = os.path.join(os.getcwd(), filename)
    pdf_obj.output(pdf_path)
    return pdf_path

# Prompt template
def build_prompt(type, resume_text, jd):
    if type == "match":
        return f"""
You are an expert ATS (Applicant Tracking System).
Evaluate the resume below against the job description and return a JSON:
{{
  "JD Match": "percentage",
  "MissingKeywords": ["list", "of", "missing", "keywords"],
  "Profile Summary": "Summary of candidate's alignment with JD"
}}

Resume:
{resume_text}

Job Description:
{jd}
"""
    elif type == "keywords":
        return f"""
Extract the missing keywords from this resume compared to the job description.

Resume:
{resume_text}

Job Description:
{jd}
"""
    elif type == "improve":
        return f"""
You are a skilled resume coach. Suggest practical improvements in bullet points to enhance the resume below for the given job description.

Resume:
{resume_text}

Job Description:
{jd}
"""
    elif type == "summary":
        return f"""
Summarize this resume in a professional tone.

Resume:
{resume_text}
"""
    elif type == "rewrite":
        return f"""
You are an expert ATS resume optimizer.
Rewrite the following resume to best match the job description while keeping it professional and well-formatted.

Resume:
{resume_text}

Job Description:
{jd}
"""

# Streamlit UI
st.set_page_config(page_title="Smart ATS", layout="centered")
st.title("🔍 Smart ATS Resume Evaluator")
jd = st.text_area("Paste the Job Description")
uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf")

if uploaded_file:
    resume_text = extract_pdf_text(uploaded_file)
    st.success("Resume uploaded successfully!")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Percentage Match"):
            prompt = build_prompt("match", resume_text, jd)
            result = llm.invoke(prompt)
            st.subheader("Percentage Match & Summary")
            st.write(result.content)

        if st.button("📋 Missing Keywords"):
            prompt = build_prompt("keywords", resume_text, jd)
            result = llm.invoke(prompt)
            st.subheader("Missing Keywords")
            st.write(result.content)

    with col2:
        if st.button("📈 Improve My Skills"):
            prompt = build_prompt("improve", resume_text, jd)
            result = llm.invoke(prompt)
            st.subheader("Improvement Suggestions")
            st.write(result.content)

        if st.button("📝 Resume Summary"):
            prompt = build_prompt("summary", resume_text, jd)
            result = llm.invoke(prompt)
            st.subheader("Resume Summary")
            st.write(result.content)

    st.markdown("---")
    if st.button("🛠️ Rewrite Resume"):
        prompt = build_prompt("rewrite", resume_text, jd)
        result = llm.invoke(prompt)
        improved_text = result.content

        pdf_path = save_text_as_pdf(improved_text, "Improved_Resume.pdf")
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download Improved Resume", f, file_name="Improved_Resume.pdf")

else:
    st.info("Please upload your resume to proceed.")
