from flask import Flask, request, jsonify
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import language_tool_python
import textstat
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

def preprocess_text(text):
    """Preprocess text: lowercase, remove punctuation, remove stopwords."""
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def compute_cosine_similarity(text1, text2):
    """Computes cosine similarity between two texts without preprocessing."""
    if not text1 or not text2:
        return 0.0

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    cosine_similarity = np.dot(vectors[0].toarray(), vectors[1].toarray().T)[0][0]
    return cosine_similarity

def extract_text_from_json(resume_data):
    """Extracts text from a JSON resume."""
    sections = []
    education = resume_data.get("education", {})
    sections.append(education.get("degree", ""))
    sections.append(education.get("university", ""))

    for exp in resume_data.get("experience", []):
        sections.append(exp.get("role", ""))
        sections.append(exp.get("organization", ""))
        sections.extend(exp.get("responsibilities", []))

    for proj in resume_data.get("projects", []):
        sections.append(proj.get("name", ""))
        sections.extend(proj.get("description", []))

    return " ".join(filter(None, sections))

def extract_skills_from_json(resume_data):
    """Extracts skills from a structured JSON resume."""
    skills = []
    technical_skills = resume_data.get("technical_skills", {})
    for skill_list in technical_skills.values():
        if isinstance(skill_list, list):
            skills.extend(skill_list)

    return " ".join(skills)

def calculate_grammar_score(text):
    """Calculates grammar score using LanguageTool."""
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    num_errors = len(matches)
    total_words = len(text.split())
    return max(0.5, 1 - (num_errors / max(1, total_words)))

def calculate_structure_score(resume_data):
    """Calculates structure score based on the presence of key sections."""
    score = 0
    total_weight = 5

    if resume_data.get("education"): 
        score += 1
    if resume_data.get("experience"): 
        score += 1
    if resume_data.get("projects"): 
        score += 1
    if resume_data.get("technical_skills"): 
        score += 1
    if resume_data.get("certifications"):  
        score += 1  

    return score / total_weight

def calculate_readability_score(text):
    """Calculates and normalizes readability score using Flesch Reading Ease."""
    raw_readability = textstat.flesch_reading_ease(text)
    scaled_score = (raw_readability + 100) / 150
    return max(0.3, min(1, scaled_score))

def calculate_vocabulary_score(text):
    """Calculates vocabulary richness using a refined type-token ratio."""
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    unique_words = set(filtered_words)
    total_words = len(filtered_words)
    
    unique_ratio = len(unique_words) / max(1, total_words)
    normalized_score = 0.5 + (0.5 * unique_ratio)

    return min(1, max(0.5, normalized_score))

def calculate_final_score(keyword_match, section_structure, formatting_compliance, readability_score, grammar_score, structure_score, vocab_score):
    """Combines all scores using the provided weight distribution."""
    weights = {
        "keyword_match": 0.5, "section_structure": 0.15, "formatting_compliance": 0.15,
        "readability": 0.2, "grammar": 0.15, "structure": 0.15, "vocab": 0.15
    }
    
    return (weights["keyword_match"] * keyword_match +
            weights["section_structure"] * section_structure +
            weights["formatting_compliance"] * formatting_compliance +
            weights["readability"] * readability_score +
            weights["grammar"] * grammar_score +
            weights["structure"] * structure_score +
            weights["vocab"] * vocab_score)

def process_resume(resume_data, job_description):
    """Processes a resume and calculates its final ATS score."""
    text = extract_text_from_json(resume_data)
    skills = extract_skills_from_json(resume_data)

    keyword_match = compute_cosine_similarity(skills, job_description)
    readability_score = calculate_readability_score(text)
    grammar_score = calculate_grammar_score(text)
    structure_score = calculate_structure_score(resume_data)
    vocab_score = calculate_vocabulary_score(text)

    section_structure = 1 if text else 0  
    formatting_compliance = 1  

    final_score = calculate_final_score(keyword_match, section_structure, formatting_compliance, readability_score, grammar_score, structure_score, vocab_score)

    return {
        "Keyword Match": keyword_match,
        "Section Structure": section_structure,
        "Formatting Compliance": formatting_compliance,
        "Readability Score": readability_score,
        "Grammar Score": grammar_score,
        "Structure Score": structure_score,
        "Vocabulary Score": vocab_score,
        "Final Score": final_score
    }

@app.route('/evaluate-resume', methods=['POST'])
def evaluate_resume():
    data = request.json
    
    job_description = data.get('job_description')
    resume_data = data.get('resume_data')
    
    if not job_description or not resume_data:
        return jsonify({"error": "Both 'job_description' and 'resume_data' are required"}), 400
    
    result = process_resume(resume_data, job_description)
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
