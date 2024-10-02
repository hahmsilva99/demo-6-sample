
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

# Function to generate response from Microsoft DialoGPT
def generate_response(input_text):
    new_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = new_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Start chatbot interaction
while True:
    # Ask user for candidate name
    candidate_name = input("Which candidate's CV would you like to query? (Harshana Madhuwantha, Isuru Aththanayake , Manesha Tharida, Buddhi Sri): ")

    if candidate_name not in cv_data:
        print("Candidate not found. Please enter a valid name.")
        continue

    # Ask for a specific query (skills, experience, education)
    user_input = input(f"You: What would you like to know about {candidate_name}? ")

    # Check if the input is related to CV details
    if any(keyword in user_input.lower() for keyword in ["name", "skills", "experience", "education"]):
        bot_response = query_cv(candidate_name, user_input)
    else:
        bot_response = generate_response(user_input)
    
    print(f"Bot: {bot_response}")