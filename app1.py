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

# Function to suggest candidates based on mentioned skills, education, or experience
def suggest_candidates(input_text):
    # Extracting data from all candidate CVs for matching
    all_criteria = {
        "skills": {skill for candidate in cv_data.values() for skill in candidate["skills"]},
        "degree": {candidate["education"]["degree"] for candidate in cv_data.values()},
        "university": {candidate["education"]["university"] for candidate in cv_data.values()},
        "graduation_year": {str(candidate["education"]["graduation_year"]) for candidate in cv_data.values()},
        "experience": {exp['position'] for candidate in cv_data.values() for exp in candidate['experience']}
    }
    
    matched_criteria = {
        "skills": [skill for skill in all_criteria["skills"] if skill.lower() in input_text.lower()],
        "degree": [degree for degree in all_criteria["degree"] if degree.lower() in input_text.lower()],
        "university": [university for university in all_criteria["university"] if university.lower() in input_text.lower()],
        "graduation_year": [year for year in all_criteria["graduation_year"] if year in input_text],
        "experience": [position for position in all_criteria["experience"] if position.lower() in input_text.lower()]
    }
    
    if any(matched_criteria.values()):
        matched_candidates = []
        for candidate_name, candidate in cv_data.items():
            # Check if candidate matches any of the extracted criteria
            if (any(skill in candidate["skills"] for skill in matched_criteria["skills"]) or
                candidate["education"]["degree"] in matched_criteria["degree"] or
                candidate["education"]["university"] in matched_criteria["university"] or
                str(candidate["education"]["graduation_year"]) in matched_criteria["graduation_year"] or
                any(exp['position'] in matched_criteria["experience"] for exp in candidate["experience"])):
                
                matched_candidates.append(candidate_name)
        
        if matched_candidates:
            return f"Based on your query, you might be interested in: {', '.join(matched_candidates)}."
        else:
            return "No candidates found with the mentioned criteria."
    else:
        return None

# Streamlit UI starts here
st.title("Candidate CV Query and Chatbot with Comprehensive Suggestions")

# Initialize conversation history
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = ""

# Text input for the user query
user_input = st.text_input("Ask about skills, education, experience, degree, university, graduation year, etc.")

# Button to submit the query
if st.button("Submit"):
    # First, check if the input matches any skills, education, or experience criteria
    suggestion = suggest_candidates(user_input)
    
    if suggestion:
        # Display candidate suggestion if there are matches
        st.session_state['conversation_history'] += f"User: {user_input}\nBot: {suggestion}\n\n"
    else:
        # Ask the user to select a candidate from the dropdown
        candidate_name = st.selectbox("Select a candidate", list(cv_data.keys()))
        
        if any(keyword in user_input.lower() for keyword in ["name", "skills", "experience", "education", "degree", "university", "graduation year"]):
            # Query the selected candidate's details
            bot_response = query_cv(candidate_name, user_input)
        else:
            # Generate a conversational response using DialoGPT
            bot_response = generate_response(user_input)
        
        # Add to the conversation history
        st.session_state['conversation_history'] += f"User: {user_input}\nBot: {bot_response}\n\n"
    
    # Display the ongoing conversation
    st.text_area("Conversation History", st.session_state['conversation_history'], height=300)
    
# Option to clear conversation history
if st.button("Clear Conversation"):
    st.session_state['conversation_history'] = ""
    st.experimental_rerun()
