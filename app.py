from flask import Flask, render_template, request
import os
import fitz  # PyMuPDF for PDFs
import docx
import requests

app = Flask(__name__)

# Temporary upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# AI Model API (Ollama)
OLLAMA_URL = "http://localhost:11434/api/generate"

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text")
    return text.strip()

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs]).strip()

def validate_resume(resume_text, job_description):
    prompt = f"""
    Analyze this resume against the job description and provide:
    1. A precise match percentage between 0-100% (just the number)
    2. Detailed feedback on:
       - Skills match (what skills match and which are missing)
       - Experience relevance (how experience aligns with requirements)
       - Project alignment (relevant projects)
       - Missing qualifications (what's lacking)
    
    Format your response like this:
    Match Score: XX%
    Skills: [analysis]
    Experience: [analysis] 
    Projects: [analysis]
    Missing Qualifications: [analysis]

    Job Description: {job_description}
    Resume Text: {resume_text}
    """

    data = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=data)
        response.raise_for_status()
        ai_response = response.json().get("response", "")

        # Extract match score
        match_score = 0
        for line in ai_response.split('\n'):
            if "Match Score:" in line:
                match_score = int(''.join(filter(str.isdigit, line.split(":")[1])))
                match_score = max(0, min(100, match_score))  # Clamp between 0-100
                break

        # Parse feedback sections
        feedback_sections = {
            'Skills': '',
            'Experience': '',
            'Projects': '',
            'Missing Qualifications': ''
        }

        current_section = None
        for line in ai_response.split('\n'):
            line = line.strip()
            if any(section in line for section in feedback_sections):
                current_section = next(section for section in feedback_sections if section in line)
                line = line.replace(current_section + ":", "").strip()
            if current_section and line:
                feedback_sections[current_section] += line + '\n'

        feedback_html = f"""
        <div class='feedback-sections'>
            <div class='section'><h3>Skills</h3><p>{feedback_sections['Skills'] or 'Not specified'}</p></div>
            <div class='section'><h3>Experience</h3><p>{feedback_sections['Experience'] or 'Not specified'}</p></div>
            <div class='section'><h3>Projects</h3><p>{feedback_sections['Projects'] or 'Not specified'}</p></div>
            <div class='section missing'><h3>Missing Qualifications</h3><p>{feedback_sections['Missing Qualifications'] or 'None identified'}</p></div>
        </div>
        """
        
        return match_score, feedback_html

    except Exception as e:
        print(f"Error processing AI response: {str(e)}")
        return 0, "<p>Error generating response from AI model. Please try again.</p>"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_description = request.form["job_description"]
        resume_file = request.files["resume"]

        if resume_file:
            file_path = os.path.join(UPLOAD_FOLDER, resume_file.filename)
            resume_file.save(file_path)

            # Extract text
            if resume_file.filename.endswith(".pdf"):
                resume_text = extract_text_from_pdf(file_path)
            elif resume_file.filename.endswith(".docx"):
                resume_text = extract_text_from_docx(file_path)
            else:
                return "Unsupported file format", 400

            # Validate resume
            match_score, feedback_html = validate_resume(resume_text, job_description)

            # Delete file after processing
            os.remove(file_path)

            return render_template("index.html", match_score=match_score, feedback_html=feedback_html)

    return render_template("index.html", match_score=None, feedback_html=None)

if __name__ == "__main__":
    app.run(debug=True)