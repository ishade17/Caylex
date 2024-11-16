import requests

from global_vars import logger, CUSTOMER_AGENT_BASE_URL, AGENT_COMM_API_BASE_URL, is_2xx_status_code, DIVISION_TAG, CENTRAL_API_BASE_URL
from send_message import auto_send_message
from present_message_with_context import present_message_with_context

# Get information about the model used before producing the message context
def get_ai_agent_info():
    # This endpoint defined in the customer_agent_api.py
    # Does this mean we need to have a docker container running for the customer_agent_api? ugh
    model_info_response = requests.get(f"{CUSTOMER_AGENT_BASE_URL}/ai_agent_info")
    if is_2xx_status_code(model_info_response.status_code):
        model_provider = model_info_response.json().get("model_provider")
        model_name = model_info_response.json().get("model_name")
        return model_provider, model_name
    else:
        logger.error("Error: did not successfully fetch ai agent info.")
        raise Exception("Error: did not successfully fetch ai agent info.")

# Call the present_message_with_context endpoint to get context
def call_present_message_with_context_endpoint(message_info, model_provider, model_name):
    # TODO: do we need to call our own endpoint or can we just call the function directly?
    # TODO: this post request is not working currently, and it is recommended to use the function directly
    # msg_context_response = requests.post(
    #     f"{AGENT_COMM_API_BASE_URL}/message/present_with_context",
    #     json={"message_info": message_info, "model_provider": model_provider, "model_name": model_name}
    # )
    # if is_2xx_status_code(msg_context_response):
    #     msg_context = msg_context_response.json().get("context")
    #     return msg_context
    # else:
    #     logger.error("Error: did not successsfully fetch message context.")
    #     raise Exception("Error: did not successsfully fetch message context.")
    
    msg_context = present_message_with_context(message_info, model_provider, model_name)
    if not msg_context:
        logger.error("Error: failed to get message context.")
        raise Exception("Error: failed to get message context.")
    return msg_context
    
# Function to call customer’s AI agent with context
def call_customer_ai_agent(prompt):
    # Call the customer’s LLM agent
    response = requests.post(f"{CUSTOMER_AGENT_BASE_URL}/ai_agent", json={"prompt": prompt})
    if is_2xx_status_code(response.status_code):
        return response.json().get("response")
    else:
        logger.error("Error: failed to call customer ai agent function.")
        raise Exception("Error: failed to call customer ai agent function.")

# Function to handle the auto-send logic once the AI responds
def call_auto_send_message_endpoint(sender_division_id, message):
    data = {
        "sender_division_id": sender_division_id,
        "message": message
    }
    # Call your own endpoint to send the message
    # TODO: do we need to call our own endpoint or can we just call the function directly?
    # response = requests.post(f"{AGENT_COMM_API_BASE_URL}/message/auto_send", json=data)
    # if not is_2xx_status_code(response.status_code):
    #     logger.error("Error: failed to auto send message.")
    #     raise Exception("Error: failed to auto send message.")

    message_info = auto_send_message(sender_division_id, message)
    if not message_info:
        logger.error("Error: failed to auto send message.")
        raise Exception("Error: failed to auto send message.")
    return message_info

def start_new_thread(message):
    sender_division_id = requests.get(f"{CENTRAL_API_BASE_URL}/divisions/tag/{DIVISION_TAG}").json()['id']
    message_info = call_auto_send_message_endpoint(sender_division_id, message)
    return message_info

# Message processing function (background task) called from receive_message()
def process_message(message_info):
    try:
        model_provider, model_name = get_ai_agent_info()

        # if not model_provider or not model_name:
        #     logger.error("Error: no model provider or model name found.")
        #     # For testing purposes...
        #     # This should be coming from the customer_agent api container
        #     agent_info = {
        #         "model_provider" : "openai",
        #         "model_name" : "GPT-4 Turbo"
        #     }
        #     model_provider = agent_info["model_provider"]
        #     model_name = agent_info["model_name"]

        msg_context = call_present_message_with_context_endpoint(message_info, model_provider, model_name)
        # msg_context = f"{msg_context} \n\n **If you think the original requested task is completed as exactly specified, then simply respond only with 'TASK COMPLETE'. If you have questions or need clarifications, then please follow-up. Iterate on the task until the request is completed as exactly specified.**"

        # Call the customer’s AI agent with the generated context
        agent_response = call_customer_ai_agent(msg_context)

        # Check if the collaboration task is complete
        # if agent_response == "TASK COMPLETE":
        #     return agent_response

        # Response to the sender division with the proper thread tag
        message_to_send = f"@thread_{message_info['thread_id']} @{message_info['sender_division_tag']} {agent_response}"

        # Automatically send the AI's response to the conversation thread
        message_info = call_auto_send_message_endpoint(
            sender_division_id=message_info["receiver_division_id"],
            message=message_to_send
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}")
