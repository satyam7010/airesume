import streamlit as st
import os
import PyPDF2
import docx
import openai
import faiss
import json
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Set OpenAI API Key
openai.api_key = "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"

# Streamlit UI
st.title("📄 AI Resume Validator & Chat")

# Upload multiple resumes
uploaded_resumes = st.file_uploader("Upload resumes (PDF/DOCX)", accept_multiple_files=True, type=["pdf", "docx"])
job_description = st.text_area("Enter Job Description")

# Extract text from resumes
def extract_text_from_resume(uploaded_file):
    text = ""
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text.strip()

# Validate Resume with OpenAI GPT
def validate_resume(resume_text, job_desc):
    system_prompt = f"Evaluate the resume against this job description:\n{job_desc}\nProvide a match score (0-100%) and detailed feedback."
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": resume_text}
        ]
    )
    feedback = response["choices"][0]["message"]["content"]
    match_score = int([int(s) for s in feedback.split() if s.isdigit()][0])
    return match_score, feedback

# Process resumes
resume_texts = {}
for resume in uploaded_resumes:
    text = extract_text_from_resume(resume)
    resume_texts[resume.name] = text

# Resume validation results
if st.button("Validate Resumes"):
    results = []
    for name, text in resume_texts.items():
        score, feedback = validate_resume(text, job_description)
        results.append({"name": name, "score": score, "feedback": feedback})
    results.sort(key=lambda x: x["score"], reverse=True)  # Sort by match score
    st.json(results)

# Create FAISS Vector Store for RAG
if st.button("Analyze with AI Chat"):
    texts = [f"{name}: {text}" for name, text in resume_texts.items()]
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.create_documents(texts)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)

    retriever = vector_store.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(model="gpt-4o")
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    # Chat UI
    user_question = st.text_input("Ask about the resumes:")
    if user_question:
        response = qa_chain.run(user_question)
        st.write(response)
