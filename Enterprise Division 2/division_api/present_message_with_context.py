import requests

from global_vars import supabase, logger, CENTRAL_API_BASE_URL, is_2xx_status_code

### PRESENT MESSAGE WITH CONTEXT ###

def format_message_chain(context_messages, division_mapping, print_statements=False):
    orig_sender_division = context_messages[0]['sender_division_id']
    orig_receiver_division = context_messages[0]['receiver_division_id']
    # todo: we should probably count the tokens for the context string as well
    context = f"-- Conversation history between the {division_mapping[orig_sender_division][1]} division at the company {division_mapping[orig_sender_division][0]} and the {division_mapping[orig_receiver_division][1]} division at the company {division_mapping[orig_receiver_division][0]} --\n\n"

    for msg_num, msg in enumerate(context_messages):
        sender_info = f"Division {division_mapping[msg['sender_division_id']][1]} said:\n"
        if msg_num == len(context_messages) - 1:
            sender_info = "-- Current message to respond to --\n\n" + sender_info
        context += f"{sender_info} {msg['message_content']}\n\n"
    return context

def fetch_thread_context(thread_id, model_provider, max_tokens, print_statements=False):
    messages = supabase.table("messages").select("*").eq("thread_id", thread_id).order("thread_msg_ordering", desc=True).execute()

    if not messages.data:
        logger.error("Error fetching thread context.")
        return None

    accumulated_tokens = 0
    context_messages = []

    for message in reversed(messages.data):
        # Get the proper way to count tokens for a given FM provider
        message_tokens = message['token_counts'].get(model_provider, 0)
        if print_statements:
            print(message)
            print(message['token_counts'])
            print(f"message_tokens: {message_tokens}")
        # Count the tokens until we reach the context window limit for the particular model
        if accumulated_tokens + message_tokens > max_tokens:
            break
        context_messages.append(message)
        accumulated_tokens += message_tokens
    if print_statements:
        print(f"accumulated_tokens: {accumulated_tokens}, max_tokens: {max_tokens}")
    return context_messages

def present_message_with_context(message_info, model_provider, model_name, print_statements=False):

    # Get context window length from central database via API call
    params = {
        'model_provider': model_provider,
        'model_name': model_name
    }
    response = requests.get(f"{CENTRAL_API_BASE_URL}/llm_context_windows", params=params)
    if is_2xx_status_code(response.status_code):
        data = response.json()
        if data:
            max_tokens = data['context_window_length']
        else:
            max_tokens = 4096  # Default value if not found
            print("Model not found in central database. Using default max tokens.")
    else:
        max_tokens = 4096  # Default value if API call fails
        logger.error(f"Error retrieving context window length: {response.status_code}. Using default max tokens.")

    # Fetch thread context from the division's own database
    thread_context = fetch_thread_context(
        message_info['thread_id'],
        model_provider,
        max_tokens=max_tokens
    )

    # Construct division mapping with company names and division tags
    division_mapping = {
        message_info['sender_division_id']: [message_info['sender_company_name'], message_info['sender_division_tag']],
        message_info['receiver_division_id']: [message_info['receiver_company_name'], message_info['receiver_division_tag']]
    }

    formatted_chain = format_message_chain(thread_context, division_mapping)
    if print_statements:
        print(formatted_chain)

    return formatted_chain