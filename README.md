# demo-6-sample
 
 Candidate CV Query and Chatbot


This project is a chatbot that allows users to query candidate CV data or ask general questions, with responses generated using Microsoft's DialoGPT. The chatbot provides information such as skills, experience, and education for predefined candidates.

Features

    Query candidate details such as name, skills, experience, and education.
    Use Microsoft DialoGPT for generating responses for non-CV related queries.
    A simple web-based UI built with Streamlit.

Prerequisites

    Python 3.8+
    pip (Python package installer)

This will install the following packages:

    transformers (for using Microsoft DialoGPT)
    torch (for model inference)
    streamlit (for building the web UI)


Usage
Start the Streamlit app:

Run the following command in your terminal or command prompt from the project directory:

 code

streamlit run streamlit_chatbot.py

This will launch the web-based chatbot interface in your default browser.

Interact with the chatbot:

    Select a candidate from the dropdown list.
    Ask questions about the selected candidate, such as skills, experience, or education.
    For non-CV-related queries, the chatbot will generate a response using DialoGPT.

Example Queries

    Skills: "What are Harshana Madhuwantha's skills?"
    Experience: "Tell me about Isuru Aththanayake's experience."
    Education: "Where did Manesha Tharida study?"

For any other inputs, the chatbot will try to generate a response using Microsoft's DialoGPT model.

Extending the Dataset

You can add more candidates to the dataset by editing the cv_data.py file. Here’s the format:

 code

cv_data = {
    "Candidate Name": {
        "name": "Full Name",
        "skills": ["Skill1", "Skill2"],
        "experience": [
            {"position": "Job Title", "company": "Company Name", "years": X}
        ],
        "education": {
            "degree": "Degree Title",
            "university": "University Name",
            "graduation_year": 20XX
        }
    }
}


Dependencies

    transformers: for working with pre-trained transformer models like DialoGPT.
    torch: required for inference with transformer models.
    streamlit: for building and running the web-based UI.
