import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from cv_data import cv_data  # Importing the CV data

# Load Microsoft DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Function to query CV data
def query_cv(candidate_name, query):
    if candidate_name in cv_data:
        cv = cv_data[candidate_name]
        
        if "name" in query.lower():
            return f"Candidate's Name: {cv['name']}"
        elif "skills" in query.lower():
            return f"Skills: {', '.join(cv['skills'])}"
        elif "experience" in query.lower():
            exp_details = "\n".join([f"{exp['position']} at {exp['company']} ({exp['years']} years)" for exp in cv['experience']])
            return f"Experience: \n{exp_details}"
        elif "education" in query.lower():
            edu = cv['education']
            return f"Education: {edu['degree']} from {edu['university']} (Graduated in {edu['graduation_year']})"
        else:
            return "I couldn't understand your query. Please ask about name, skills, experience, or education."
    else:
        return "Candidate not found. Please provide a valid name."

# Function to generate a response from Microsoft DialoGPT
def generate_response(input_text):
    new_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = new_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Function to suggest candidates based on mentioned skills
def suggest_candidates(input_text):
    # List of all skills in the CV dataset
    all_skills = {skill for candidate in cv_data.values() for skill in candidate["skills"]}
    
    # Extract matching skills from user input
    mentioned_skills = [skill for skill in all_skills if skill.lower() in input_text.lower()]
    
    if mentioned_skills:
        # Find candidates with matching skills
        matched_candidates = []
        for candidate_name, candidate in cv_data.items():
            if any(skill in candidate["skills"] for skill in mentioned_skills):
                matched_candidates.append(candidate_name)
        
        if matched_candidates:
            return f"Based on your interest in {', '.join(mentioned_skills)}, you might want to check out: {', '.join(matched_candidates)}."
        else:
            return "No candidates found with the mentioned skills."
    else:
        return None

# Streamlit UI starts here
st.title("Candidate CV Query and Chatbot with Skill Suggestion")

# Text input for the user query
user_input = st.text_input("Ask about skills, experience, education, or programming languages.")

# Button to submit the query
if st.button("Submit"):
    # First, check if the input is related to skills and suggest candidates
    skill_suggestion = suggest_candidates(user_input)
    
    if skill_suggestion:
        # If there are skill matches, suggest candidates
        st.write(f"Bot: {skill_suggestion}")
    else:
        # Otherwise, ask the user to select a candidate from the dropdown
        candidate_name = st.selectbox("Select a candidate", list(cv_data.keys()))
        
        if any(keyword in user_input.lower() for keyword in ["name", "skills", "experience", "education"]):
            # If the query is related to CV details
            bot_response = query_cv(candidate_name, user_input)
        else:
            # Otherwise, generate a conversational response using DialoGPT
            bot_response = generate_response(user_input)
        
        # Display the response
        st.write(f"Bot: {bot_response}")
