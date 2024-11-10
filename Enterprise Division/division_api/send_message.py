import requests
import re
import hashlib

from datetime import datetime

from global_vars import logger, supabase, openai_client, CENTRAL_API_BASE_URL, is_2xx_status_code, COMPANY_NAME
from count_message_tokens import default_models, tokenizers
from data_policy_rag import retrieve_relevant_document_chunks
from database_interaction import find_connection, get_division_api_url

### SEND MESSAGES ###

def check_policy_compliance(message, policy, from_doc=False, print_statements=False):
    if from_doc:
        policy_text = f"Data policy excerpt: '{policy}'"
    else:
        policy_text = f"Policy: \n'{policy['natural_language_explanation']}'"

    input = f"""Review this message: '{message}'

                {policy_text}

                Does this message comply with the policy?
                If so, answer with only 'Yes, compliant.'
                If not, answer with 'No, not compliant.' Then explain why."""
    if print_statements:
        print(input)
    try:
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": input,
                }
            ],
            model="gpt-4",
            temperature=0
        )
        compliance_response = response.choices[0].message.content.strip()
        if print_statements:
            print(f"compliance_response: {compliance_response}")
            print(f"compliant? {'yes, compliant' in compliance_response.lower()}")
        return "yes, compliant" in compliance_response.lower(), compliance_response
    except Exception as e:
        logger.error(f"Error occurred while checking policy compliance: {e}")
        return False, str(e)

def log_message(sender_division_id, receiver_division_id, connection, local_connection_id, raw_api_key, message, thread_id="new thread", print_statements=False):
    # Calculate token counts
    token_counts = {}
    for provider, default_model in default_models.items():
        count_function = tokenizers[provider]
        token_counts[provider] = count_function(message, default_model)
    if print_statements:
        print(f"token_counts in log_message(): {token_counts}")

    if thread_id == "new thread":
        # Create a new thread in the central database via API call
        # since we are inserting into the central database, we use the central_connection_id (i.e., connection['id'])
        thread_entry = {
            "connection_id": connection['id'],
            "last_message_timestamp": datetime.now().isoformat(),
            "source_division_cost": 0.0001 if sender_division_id == connection['source_division_id'] else 0,
            "target_division_cost": 0.0001 if sender_division_id == connection['target_division_id'] else 0,
            "messages_count": 1
        }
        create_thread_response = requests.post(f"{CENTRAL_API_BASE_URL}/threads", json=thread_entry)
        if is_2xx_status_code(create_thread_response.status_code):
            thread_data = create_thread_response.json()
            thread_id = thread_data['thread_id']
            thread_msg_ordering = 1
        else:
            logger.error("Error creating thread in central database.")
            return None, None, None, None
    else:
        # Update existing thread in the central database via API call
        response = requests.get(f"{CENTRAL_API_BASE_URL}/threads/{thread_id}")
        if is_2xx_status_code(response.status_code):
            thread_data = response.json()
            if not thread_data:
                print("Thread not found in central database.")
                return None, None, None, None
            update_data = {
                "messages_count": thread_data['messages_count'] + 1,
                "last_message_timestamp": datetime.now().isoformat(),
                "source_division_cost": thread_data['source_division_cost'] + (0.0001 if sender_division_id == connection['source_division_id'] else 0),
                "target_division_cost": thread_data['target_division_cost'] + (0.0001 if sender_division_id == connection['target_division_id'] else 0)
            }
            update_thread_response = requests.put(f"{CENTRAL_API_BASE_URL}/threads/{thread_id}", json=update_data)
            if is_2xx_status_code(update_thread_response.status_code):
            # if not update_thread_response:
                logger.error("Error updating thread in central database.")
                return None, None, None, None
            thread_msg_ordering = thread_data['messages_count'] + 1
        else:
            logger.error("Error accessing thread in central database.")
            return None, None, None, None
        
    # since we are inserting into the local database, we use the local_connection_id
    message_entry = {
        "connection_id": local_connection_id,
        "sender_division_id": sender_division_id,
        "receiver_division_id": receiver_division_id,
        "message_content": message,
        "timestamp": datetime.now().isoformat(),
        "status": "sent",
        "thread_id": thread_id,
        "thread_msg_ordering": thread_msg_ordering,
        "token_counts": token_counts
    }
    if print_statements:
        print(f"message_entry in log_message(): {message_entry}")

    # Insert the message into the sender's local database
    sender_result = supabase.table("messages").insert(message_entry).execute()
        
    # Send API request to receiver division to log the message in their database
    receiver_api_url = get_division_api_url(receiver_division_id)
    if not receiver_api_url:
        print("Cannot find receiver division's API URL.")
        return None, None, None, None

    # Retrieve the API key for the connection (assuming it's available)
    # api_key = connection['api_key']
    if not raw_api_key:
        logger.error("No API Key found for the connection.")
        return None, None, None, None
    
    # Prepare the data to send
    data_to_send = {
        "message_entry": message_entry
    }

    headers = {
        "X-API-KEY": raw_api_key
    }

    response = requests.post(f"{receiver_api_url}/messages/receive", json=data_to_send, headers=headers)
    if is_2xx_status_code(response.status_code):
        receiver_response = response.json()
        receiver_message_id = receiver_response.get('message_id')
    else:
        logger.error(f"Error sending message to receiver division: {response.status_code}")
        return None, None, None, None

    return sender_result.data[0]['message_id'], receiver_message_id, thread_id, thread_msg_ordering

def get_raw_hashed_api_keys(sender_division_id, receiver_division_id):
    filter_string = (
        f"and(source_division_id.eq.{sender_division_id},target_division_id.eq.{receiver_division_id}),"
        f"and(source_division_id.eq.{receiver_division_id},target_division_id.eq.{sender_division_id})"
    )
    # Query connections where the two divisions are connected, regardless of order
    result = supabase.table("connections").select("raw_api_key").or_(filter_string).execute()
    raw_api_key = result.data[0]['raw_api_key']
    hashed_api_key = hashlib.sha256(raw_api_key.encode()).hexdigest()
    return raw_api_key, hashed_api_key

def send_message(sender_division_id, receiver_division_id, message, thread_id="new thread", print_statements=False):
    # Find connection via API call
    raw_api_key, hashed_api_key = get_raw_hashed_api_keys(sender_division_id, receiver_division_id)

    # TODO: big problem here. we are retuning the central_connection_id, but all the data tables reference the local_connection_id.
    # update: done.
    connection, approved_to_send_message, local_connection_id = find_connection(sender_division_id, receiver_division_id, hashed_api_key)

    if not connection:
        print("No valid connection found.")
        raise Exception("No valid connection found for this proposed message.")
        # return False, None, None, None, None, None
    if not approved_to_send_message:
        print("Message not approved to be sent.")
        raise Exception("Message not approved to be sent.")

    # Retrieve data policies from local database
    policies = supabase.table("custom_data_policies").select("*").eq("connection_id", local_connection_id).execute()

    # Policy compliance checks
    if policies.data:
        for policy in policies.data:
            compliant_bool, compliance_response = check_policy_compliance(message, policy)
            if not compliant_bool:
                print(f"Message violates policy: \n{policy['natural_language_explanation']} \n\nCompliance response: \n{compliance_response}")
                return False, None, None, None, None, None
    else:
        print("No data policies found for this connection. Proceeding with message sending.")

    # Retrieve relevant document chunks
    # TODO: we could just pass in the local_connection_id instead of the connection object? done.
    relevant_chunks = retrieve_relevant_document_chunks(message, local_connection_id, top_k=2)
    if relevant_chunks:
        for chunk in relevant_chunks:
            compliant_bool, compliance_response = check_policy_compliance(message, chunk['content'], from_doc=True)
            if not compliant_bool:
                print(f"Message violates a policy from this excerpt: \n{chunk['content']} \n\nCompliance response: \n{compliance_response}")
                return False, None, None, None, None, None
    else:
        print("No relevant document chunks found. Proceeding with message sending.")

    # Log message
    # TODO: we could just pass in the local_connection_id instead of the connection object?
    # we need to pass in both.
    sender_message_id, receiver_message_id, thread_id, thread_msg_ordering = log_message(
        sender_division_id, receiver_division_id, connection, local_connection_id, raw_api_key, message, thread_id
    )

    if print_statements:
        print(f"Message sent successfully. Sender message ID: {sender_message_id}, Receiver message ID: {receiver_message_id}, Thread ID: {thread_id}, Message Ordering: {thread_msg_ordering}")

    return True, sender_message_id, receiver_message_id, thread_id, thread_msg_ordering, connection

def auto_send_message(sender_division_id, message, print_statements=False):
    # Regex patterns to find tags
    division_tag_pattern = re.compile(r"@(?!thread_|new_thread)([a-zA-Z0-9_]+)")
    thread_tag_pattern = re.compile(r"@thread_(\d+)")
    new_thread_tag_pattern = re.compile(r"@new_thread")

    # Extract tags from the message
    division_tags = division_tag_pattern.findall(message)
    thread_tags = thread_tag_pattern.findall(message)
    new_thread_tags = new_thread_tag_pattern.findall(message)

    # Determine thread ID
    if new_thread_tags:
        thread_id = "new thread"
    elif thread_tags:
        if len(thread_tags) > 1:
            logger.error("Error: Multiple thread tags detected. Please specify only one.")
            return None
        thread_id = int(thread_tags[0])
    else:
        thread_id = None

    if len(division_tags) != 1:
        logger.error("Error: Please specify exactly one division tag.")
        return None

    # Get receiver division ID from the tag
    receiver_division_tag = division_tags[0]
    # Query central database via API to get receiver division details
    receiver_tag_response = requests.get(f"{CENTRAL_API_BASE_URL}/divisions/tag/{receiver_division_tag}")
    if is_2xx_status_code(receiver_tag_response.status_code):
        division_info = receiver_tag_response.json()
        receiver_division_id = division_info['id']
        receiver_company_id = division_info['company_id']
        receiver_division_tag = division_info['tag']

        # Get receiver company name
        receiver_response_company = requests.get(f"{CENTRAL_API_BASE_URL}/companies/{receiver_company_id}")
        if is_2xx_status_code(receiver_response_company.status_code):
            receiver_company_info = receiver_response_company.json()
            receiver_company_name = receiver_company_info['name']
        else:
            logger.error("Error retrieving receiver company information.")
            return None
    else:
        logger.error("Error: No division found with the specified tag.")
        return None

    # Query central database via API to get sender division info
    sender_response_division = requests.get(f"{CENTRAL_API_BASE_URL}/divisions/{sender_division_id}")
    if is_2xx_status_code(sender_response_division.status_code):
        division_info = sender_response_division.json()
        sender_division_tag = division_info['tag']
        sender_company_id = division_info['company_id']

        # Get sender company name
        # TODO: we may just want to pass in sender_company_name to the function like we were doing before?
        # TODO: we already have COMPANY_NAME as an env variable, so i think this api call is unnecessary
        # sender_response_company = requests.get(f"{CENTRAL_API_BASE_URL}/companies/{sender_company_id}")
        # if is_2xx_status_code(sender_response_company.status_code):
        #     sender_company_info = sender_response_company.json()
        #     sender_company_name = sender_company_info['name']
        # else:
        #     logger.error("Error retrieving sender company information.")
        #     return None
    else:
        logger.error("Error retrieving sender division information.")
        return None

    # Clean message by removing tags
    clean_message = re.sub(r"@[\w_]+", "", message).strip()

    if print_statements:
        print(f"Sender Company: {COMPANY_NAME} (Division ID: {sender_division_id}, Tag: {sender_division_tag})")
        print(f"Receiver Company: {receiver_company_name} (Division ID: {receiver_division_id}, Tag: {receiver_division_tag})")
        print(f"Thread ID: {thread_id if thread_id else 'new thread'}")
        print(f"Message: {clean_message}")

    # Send the message
    compliance_pass, sender_message_id, receiver_message_id, thread_id, thread_msg_ordering, connection = send_message(
        sender_division_id,
        receiver_division_id,
        clean_message,
        thread_id if thread_id else "new thread"
    )

    if not compliance_pass:
        return None

    message_info = {
        "sender_company_name": COMPANY_NAME,
        "receiver_company_name": receiver_company_name,
        "sender_division_id": sender_division_id,
        "receiver_division_id": receiver_division_id,
        "sender_division_tag": sender_division_tag,
        "receiver_division_tag": receiver_division_tag,
        "message": clean_message,
        "thread_id": thread_id,
        "thread_msg_ordering": thread_msg_ordering,
        "connection_id": connection['id']
    }

    return message_info