import os

from flask import Flask, request, jsonify

from openai import OpenAI
import google.generativeai as genai

app = Flask(__name__)

# theoretically, these will be the customer's api keys
# we don't need the parent directory because the .env file is in the same folder right now
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_KEY"),
)
genai.configure(api_key=os.getenv("GOOGLE_KEY"))


### SAMPLE AI AGENTS ###

# Example 1
def openai_agent(prompt, system_prompt="You are a helpful assistant."):
    system_prompt = "You are a helpful assistant that specializing in working with companies in the finance industry to build custom data integrations with Bloomberg for their data science, risk management, and trading teams. Your communication style should be very logical and concise, and you are very results-driven. You should produce tangible work products."
    agent_response = openai_client.chat.completions.create(
            messages=[
                { 
                    "role" : "system", 
                    "content" : system_prompt
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4-turbo",
            temperature=0
        ).choices[0].message.content
    return agent_response

# Example 2
def gemini_agent(prompt, system_prompt="You are a helpful assistant."):
    agent_response = genai.GenerativeModel("gemini-1.5-pro-001").generate_content(prompt).text
    return agent_response


### NECESSARY ENDPOINTS TO EXPOSE FOR CUSTOMERS ###

@app.route('/ai_agent', methods=['POST'])
def ai_agent():
    prompt = request.get_json().get("prompt")

    # Call your agent's function here...
    response = openai_agent(prompt)

    return jsonify({"response" : response})

@app.route('/ai_agent_info', methods=['GET'])
def ai_agent_info():

    # Insert your agent's info here...
    agent_info = {
        "model_provider" : "openai",
        "model_name" : "GPT-4 Turbo"
    }
    
    return agent_info


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)



# openai_agent = {
#     "sender_company_name" : "bloomberg",
#     "sender_division_id" : 4,
#     "model_type" : "openai",
#     "model_name" : "GPT-4 Turbo",
#     "system_prompt" : "You are a helpful assistant that specializing in working with companies in the finance industry to build custom data integrations with Bloomberg for their data science, risk managment, and trading teams. Your communication style should be very logical and concise, and you are very results-driven. You should produce tangible work products."
# }

# gemini_agent = {
#     "sender_company_name" : "global financial corp",
#     "sender_division_id" : 1,
#     "model_type" : "google",
#     "model_name" : "Gemini 1.5 Pro",
#     "system_prompt" : "You are a helpful assistant that specializing in developing alternative data solutions for investors and traders. Your communication style should be very logical and concise, and you are very results-driven. You should produce tangible work products."
# }

