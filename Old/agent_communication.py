import os
import re
import fitz
import faiss
import torch
import tiktoken

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import google.generativeai as genai

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pprint import pprint
from datetime import datetime
from supabase import create_client, Client
from pyvis.network import Network
from IPython.display import HTML, display
from vertexai.preview import tokenization

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from semantic_router.splitters import RollingWindowSplitter
from semantic_router.utils.logger import logger
from semantic_router.encoders import OpenAIEncoder

from transformers import LlamaTokenizer
from transformers import AutoTokenizer
from transformers import T5Tokenizer


### API KEYS ###

# Connect to Supabase
url: str = "https://rtzkvrxmbdwyydercpzh.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ0emt2cnhtYmR3eXlkZXJjcHpoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjEwMTY3MDcsImV4cCI6MjAzNjU5MjcwN30.waykwg4OMiDpT0TNU-95dP45oxhhdv8T6rz8yTy0cNo"
supabase: Client = create_client(url, key)


openai_api_key = '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'
os.environ['OPENAI_API_KEY'] = openai_api_key

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

google_api_key = "AIzaSyATwXkB_Nngdqq1rW5Hna9cwOhJM26YWcA"
os.environ['GOOGLE_API_KEY'] = google_api_key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


### DATABASE SCHEMA ###

def register_company(name):
    data = {"name": name}
    result = supabase.table("companies").insert(data).execute()
    return result.data[0]['id']

def register_division(company_id, name, tag):
    data = {"company_id": company_id, "name": name, "tag": tag}
    print(data)
    result = supabase.table("divisions").insert(data).execute()
    return result.data[0]['id']

def add_connection(source_division_id, target_division_id, daily_messages_count=0):
    data = {"source_division_id": source_division_id, "target_division_id": target_division_id, "daily_messages_count": daily_messages_count}
    result = supabase.table("connections").insert(data).execute()
    return result.data[0]['id']

def find_connection(sender_division_id, receiver_division_id):
    # Query for the connection in both possible directions
    query1 = supabase.table("connections").select("*").eq("source_division_id", sender_division_id).eq("target_division_id", receiver_division_id).execute()
    query2 = supabase.table("connections").select("*").eq("source_division_id", receiver_division_id).eq("target_division_id", sender_division_id).execute()

    if query1.data:
        return query1.data[0]  # Return the first result if available
    elif query2.data:
        return query2.data[0]  # Return the first result from the second query if available
    else:
        return None  # No connection found

def insert_data_policy(sender_division_id, receiver_division_id, confidentiality, data_type, explanation):
    # Look up the connection ID using sender and receiver division IDs
    print(sender_division_id, receiver_division_id)
    connection = find_connection(sender_division_id, receiver_division_id)

    if not connection:
        return "No valid connection found between the specified divisions."

    connection_id = connection['id']
    print(f"Data policy for connection #{connection_id}: {explanation}")

    # Determine which company owns the sending division to decide where to store the policy
    company_info = supabase.table("divisions").select("company_id").eq("id", sender_division_id).execute()
    if not company_info.data:
        return "Sending division not found."
    company_id = company_info.data[0]['company_id']

    # Retrieve the company name based on company_id for table naming
    company = supabase.table("companies").select("name").eq("id", company_id).execute()
    if not company.data:
        return "Company not found."
    company_name = company.data[0]['name']

    # Format the table name from the company name
    table_name = format_company_table_name(company_name, "data_policies")

    # Insert data into the company-specific data policies table
    data = {
        "connection_id": connection_id,
        "confidentiality": confidentiality,
        "data_type": data_type,
        "natural_language_explanation": explanation
    }
    result = supabase.table(table_name).insert(data).execute()
    return result.data

def get_data_policies(company_id, division_id=None, receiving_division_id=None):
    # Retrieve the company name based on company_id to determine the table name
    company_info = supabase.table("companies").select("name").eq("id", company_id).execute()
    if not company_info.data:
        return "Company not found"
    company_name = company_info.data[0]['name']
    table_name = format_company_table_name(company_name, "data_policies")

    # Build the query based on the level of specificity required
    query = supabase.table(table_name).select("*")

    if division_id:
        # Fetch connections for the specific division, either as source or target
        connections = supabase.table("connections").select("id").or_(
            f"source_division_id.eq.{division_id},target_division_id.eq.{division_id}"
        ).execute()
        if not connections.data:
            return "No connections found for the specified division"
        connection_ids = [conn['id'] for conn in connections.data]
        query = query.in_("connection_id", connection_ids)
    elif receiving_division_id:
        # Fetch policies for all company divisions connected to a specified receiving division
        connections = supabase.table("connections").select("id").or_(
            f"source_division_id.eq.{receiving_division_id},target_division_id.eq.{receiving_division_id}"
        ).execute()
        if not connections.data:
            return "No connections found for the specified receiving division"
        connection_ids = [conn['id'] for conn in connections.data]
        query = query.in_("connection_id", connection_ids)

    # Execute the query and return the results
    result = query.execute()
    return result.data if result.data else "No policies found"

def format_company_table_name(company_name, table_type=None):
    # Remove commas and periods, replace spaces with underscores, convert to lowercase
    formatted_name = company_name.replace(",", "").replace(".", "").replace(" ", "_").lower()
    if table_type:
        table_type_name = f"_{table_type}"
    else:
        table_type_name = ""
    return f"{formatted_name}{table_type_name}"

def get_threads_table_name(connection_id):
    return f"collab_threads_{connection_id}"


### SET UP SAMPLE NETWORK ###
# TODO: this should really be in a separate script

def setup_complex_network():
    return None


### RAG FOR DATA POLICIES ###

encoder = OpenAIEncoder(name="text-embedding-ada-002")

logger.setLevel("WARNING")  # reduce logs from splitter

splitter = RollingWindowSplitter(
    encoder=encoder,
    dynamic_threshold=True,
    min_split_tokens=75,
    max_split_tokens=500,
    window_size=2,
    plot_splits=False,  # set this to true to visualize chunking
    enable_statistics=True  # to print chunking stats
)

def generate_embeddings(text, print_statements=False):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    if print_statements:
        print(response)
    return response.data[0].embedding

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def build_text_chunk(company_name, content, print_chunk=False):
    chunk = f"--- The following passage is an extract from a data policies document for {company_name} ---\n\n {content}"
    if print_chunk:
        print(chunk)
    return chunk

def store_policy_chunks(text_chunks, confidentiality, data_type, company_name, document_number, applicable_connections):
    table_name = format_company_table_name(company_name, "data_policy_docs")
    link_table_name = format_company_table_name(company_name, "document_connections")

    for chunk_order, chunk in enumerate(text_chunks):
        embedding = generate_embeddings(chunk)
        # Insert chunk into the data_policy_docs table
        result = supabase.table(table_name).insert({
            "data_type": data_type,
            "content": chunk,
            "embedding": embedding,
            "document_num": document_number,
            "chunk_order": chunk_order
        }).execute()

        # if result.error:
        #     print("Error inserting data chunk:", result.error.message)
        #     continue

        # Retrieve the document_chunk_id from the insertion result
        document_chunk_id = result.data[0]['id']

        # Insert connections for this document chunk
        for connection_id in applicable_connections:
            connection_result = supabase.table(link_table_name).insert({
                "document_chunk_id": document_chunk_id,
                "connection_id": connection_id
            }).execute()

            # if connection_result.error:
            #     print("Error linking document chunk to connection:", connection_result.error.message)

def split_store_chunks(data_policy_doc_infos):
    for company_name, infos in data_policy_doc_infos.items():
        for doc_num, info in enumerate(infos):
            location = info['location']
            applicable_connections = info['applicable_connections']
            text = extract_text_from_pdf(location)
            splits = splitter([text])
            policy_chunks = [build_text_chunk(company_name, split.content, True) for split in splits]
            store_policy_chunks(policy_chunks, "high", "financial data", company_name, doc_num, applicable_connections)

# TODO: these should probably be in a data table
# TODO: we should also add a function for registering a data policy doc
data_policy_doc_infos = {
    "Bloomberg" : [
        {"location": "/Data Policies/Bloomberg Data Policies.pdf",
         "applicable_connections" : [1, 4]
        }
    ],
    "Global Financial Corp" : [
        {"location": "/Data Policies/Global Financial Corp Data Policies.pdf",
         "applicable_connections" : [1, 2]
        }
    ]
}

def retrieve_relevant_document_chunks(query, company_name, connection, threshold=0.7, top_k=5, sender_division_id=None, receiver_division_id=None):
    # For testing in isolation
    if not connection:
        if sender_division_id and receiver_division_id:
            connection = find_connection(sender_division_id, receiver_division_id)
        else:
            return "No valid connection found or provided."

    # Generate query embedding
    query_embedding = generate_embeddings(query)

    # Prepare the embedding for database compatibility, typically converting a tensor to a list of floats
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()
    elif isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.numpy().tolist()

    # Get the IDs of document chunks applicable to this connection
    document_connection_table = format_company_table_name(company_name, "document_connections")
    document_chunk_ids = supabase.from_(document_connection_table).select("document_chunk_id").eq("connection_id", connection['id']).execute()
    # if document_chunk_ids.error:
    #     print("Error fetching document chunk IDs:", document_chunk_ids.error.message)
    #     return None
    document_chunk_ids = [item['document_chunk_id'] for item in document_chunk_ids.data]

    # Now invoke the RPC function to get relevant document chunks based on embeddings
    table_name = format_company_table_name(company_name, "data_policy_docs")
    result = supabase.rpc('match_document_chunks', {
        'table_name': table_name,
        'document_chunk_ids': document_chunk_ids,
        'query_embedding': query_embedding,
        'match_threshold': threshold,
        'match_count': top_k
    }).execute()

    # if result.error:
    #     print("Error during retrieval:", result.error.message)
    #     return None

    return result.data


### COUNT MESSAGE TOKENS ###

# Define tokenizer functions
def count_tokens_openai(text, model):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def count_tokens_meta(text, model):
    tokenizer = LlamaTokenizer.from_pretrained(model)
    return len(tokenizer.encode(text))

def count_tokens_anthropic(text, model):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def count_tokens_google(text, model):
    tokenizer = tokenization.get_tokenizer_for_model(model)
    result = tokenizer.count_tokens(text)
    return result.total_tokens

def count_tokens_mistral(text, model):
    tokenizer = MistralTokenizer.from_model(model)
    chat_request = ChatCompletionRequest(messages=[UserMessage(content=text)], model=model)
    tokenized = tokenizer.encode_chat_completion(chat_request)
    return len(tokenized.tokens)

def count_tokens_cohere(text, model):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def count_tokens_with_auto_tokenizer(text, model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    return len(tokenizer.encode(text))

# Default models for various LLM providers
# TODO: maybe also throw this in a data table
default_models = {
    'openai': 'gpt-4-turbo',
    # 'meta': 'meta-llama/Llama-2-7b-hf',
    # 'anthropic': 'claude-3',
    'anthropic': 'gpt-4-turbo',
    'google': 'gemini-1.5-pro-001',
    'mistral': 'open-mixtral-8x22b',
    # 'cohere': 'cohere-command'
    'cohere': 'gpt-4-turbo'
}

# Tokenizers dictionary with a simplified lambda that supports passing a model name
tokenizers = {
    'openai': count_tokens_openai,
    'meta': count_tokens_meta,
    'anthropic': count_tokens_anthropic,
    'google': count_tokens_google,
    'mistral': count_tokens_mistral,
    'cohere': count_tokens_cohere,
}

# Currently is unused:
# Function to count tokens that now handles an optional model_name
def count_tokens(text, model_type, model_name=None):
    # Fetch the default model if no specific model_name is provided
    if not model_name:
        model_name = default_models.get(model_type)
    if not model_name:
        raise ValueError("No model specified and no default model available for this type.")
    
    # Dynamically select and call the tokenizer function
    tokenizer_function = tokenizers.get(model_type)
    if tokenizer_function:
        return tokenizer_function(text, model_name)
    else:
        raise ValueError("Unsupported model type provided for tokenization.")

# TODO: maybe add a column for the official name of the mdoel as it is stored in the default models dict? not sure if this is really needed though because this is really just a look up table and does not interact with the actual counting of tokens
# TODO: when is this used?
def insert_context_window_length(provider, model, length): 
    result = supabase.table("llm_context_windows").insert({
        "model_provider": provider,
        "model_name": model,
        "context_window_length": length
    }).execute()

    # if result.error:
    #     print(f"Error inserting data for {model}: {result.error.message}")


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
        response = client.chat.completions.create(
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
        print(f"An error occurred: {e}")
        return False, str(e)

def log_message(sender_company_name, receiver_company_name, connection, sender_division_id, receiver_division_id, message, thread_id="new thread", print_statements=False):
    sender_table_name = format_company_table_name(sender_company_name, "messages")
    receiver_table_name = format_company_table_name(receiver_company_name, "messages")
    threads_table_name = get_threads_table_name(connection['id'])

    # Calculate token counts for all model providers 
    token_counts = {}
    for provider, default_model in default_models.items():
        count_function = tokenizers[provider]
        token_counts[provider] = count_function(message, default_model)
    if print_statements:
        print(f"token_counts in log_message(): {token_counts}")

    if isinstance(thread_id, str):
        if thread_id.lower() == "new thread":
            thread_entry = {
                "connection_id": connection['id'],
                "last_message_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_division_cost": 0.0001 if sender_division_id == connection['source_division_id'] else 0,
                "target_division_cost": 0.0001 if sender_division_id == connection['target_division_id'] else 0,
                "messages_count": 1
            }
            thread_result = supabase.table(threads_table_name).insert(thread_entry).execute()
            thread_id = thread_result.data[0]['thread_id']
            thread_msg_ordering = 1
        else:
            print("Invalid thread_id or new thread.")
            return None, None, None, None
    else:
        # Existing thread handling
        thread = supabase.table(threads_table_name).select("*").eq("thread_id", thread_id).execute().data[0]
        update_data = {
            "messages_count": thread['messages_count'] + 1,
            "last_message_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_division_cost": thread['source_division_cost'] + (0.0001 if sender_division_id == connection['source_division_id'] else 0),
            "target_division_cost": thread['target_division_cost'] + (0.0001 if sender_division_id == connection['target_division_id'] else 0)
        }
        supabase.table(threads_table_name).update(update_data).eq("thread_id", thread_id).execute()
        thread_msg_ordering = thread['messages_count'] + 1

    message_entry = {
        "connection_id": connection['id'],
        "sender_division_id": sender_division_id,
        "receiver_division_id": receiver_division_id,
        "message_content": message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "sent",
        "thread_id": thread_id,
        "thread_msg_ordering": thread_msg_ordering,
        "token_counts": token_counts
    }
    if print_statements:
        print(f"message_entry in log_message(): {message_entry}")
    sender_result = supabase.table(sender_table_name).insert(message_entry).execute()
    receiver_message_entry = message_entry.copy()
    receiver_message_entry['status'] = "received"
    receiver_result = supabase.table(receiver_table_name).insert(receiver_message_entry).execute()

    return sender_result.data[0]['message_id'], receiver_result.data[0]['message_id'], thread_id, thread_msg_ordering

def send_message(sender_company_name, receiver_company_name, sender_division_id, receiver_division_id, message, thread_id="new thread", print_statements=False):

    # Retrieve company-specific connection using sender and receiver names
    connection = find_connection(sender_division_id, receiver_division_id)
    if not connection:
        return "No valid connection found."

    data_policies_table_name = format_company_table_name(sender_company_name, "data_policies")

    # Retrieve data policies for the connection
    policies = supabase.table(data_policies_table_name).select("*").eq("connection_id", connection['id']).execute()

    if policies.data:
        for policy in policies.data:
            compliant_bool, compliance_response = check_policy_compliance(message, policy)
            if not compliant_bool:
                print(f"Message violates policy: \n{policy['natural_language_explanation']} \n\nCompliance response: \n{compliance_response}")
                return False, None, None, None, None, None
    else:
        print("No data policies found for this connection. Proceeding with message sending.")

    relevant_chunks = retrieve_relevant_document_chunks(message, sender_company_name, connection, top_k=2)
    if relevant_chunks:
        for chunk in relevant_chunks:
            compliant_bool, compliance_response = check_policy_compliance(message, chunk['content'], from_doc=True)
            if not compliant_bool:
                print(f"Message violates a policy from this excerpt: \n{chunk['content']} \n\nCompliance response: \n{compliance_response}")
                return False, None, None, None, None, None
    else:
        print("No relevant document chunks found. Proceeding with message sending.")

    # Log and update the message thread
    sender_message_id, receiver_message_id, thread_id, thread_msg_ordering = log_message(
        sender_company_name, receiver_company_name, connection, sender_division_id, receiver_division_id, message, thread_id
    )

    if print_statements:
        print(f"Message sent successfully. Sender message ID: {sender_message_id}, Receiver message ID: {receiver_message_id}, Thread ID: {thread_id}, Message Ordering: {thread_msg_ordering}")

    return True, sender_message_id, receiver_message_id, thread_id, thread_msg_ordering, connection

def auto_send_message(sender_company_name, sender_division_id, message, print_statements=False):
    # Regex patterns to find tags
    division_tag_pattern = re.compile(r"@(?!thread_|new_thread)([a-zA-Z0-9_]+)")
    thread_tag_pattern = re.compile(r"@thread_(\d+)")
    new_thread_tag_pattern = re.compile(r"@new_thread")

    # Extract tags from the message
    division_tags = division_tag_pattern.findall(message)
    thread_tags = thread_tag_pattern.findall(message)
    new_thread_tags = new_thread_tag_pattern.findall(message)

    # Check for explicit new thread tag
    if new_thread_tags:
        thread_id = "new thread"
    elif thread_tags:
        if len(thread_tags) > 1:
            print("Error: Multiple thread tags detected. Please specify only one.")
            return
        thread_id = int(thread_tags[0])
    else:
        thread_id = None

    if len(division_tags) > 1:
        print("Error: Multiple division tags detected. Please specify only one.")
        return None

    if not division_tags and not thread_id:
        print("No appropriate tags found. No message sent.")
        return None

    # Find the receiving division and its corresponding company if a division tag is present
    if division_tags:
        receiver_division_tag = division_tags[0]
        division_info = supabase.table("divisions").select("id", "company_id").eq("tag", receiver_division_tag).execute()
        if not division_info.data:
            print("Error: No division found with the specified tag.")
            return None
        receiver_division_id = division_info.data[0]['id']
        receiver_company_info = supabase.table("companies").select("name").eq("id", division_info.data[0]['company_id']).execute()
        receiver_company_name = receiver_company_info.data[0]['name']
    else:
        print("Division tag is required to send a message.")
        return None

    # Check for a valid connection
    connection = find_connection(sender_division_id, receiver_division_id)
    if not connection:
        print("No valid connection found between specified divisions.")
        return None

    sender_division_tag = supabase.table("divisions").select("tag").eq("id", sender_division_id).execute().data[0]['tag']

    # Validate the thread if specified
    if thread_id and isinstance(thread_id, int):  # Ensure thread_id is an integer for existing threads
        threads_table_name = get_threads_table_name(connection['id'])
        thread_info = supabase.table(threads_table_name).select("*").eq("thread_id", thread_id).execute()
        if not thread_info.data:
            print(f"No thread found with ID {thread_id} in the specified connection.")
            return None

    # Remove tags from the message before sending
    clean_message = re.sub(r"@[\w_]+", "", message).strip()

    if print_statements:
        print(f"Sender: {sender_company_name} (Division ID: {sender_division_id})")
        print(f"Receiver: {receiver_company_name} (Division ID: {receiver_division_id})")
        print(f"Thread ID: {thread_id if thread_id else 'new thread'}")
        print(f"Message: {clean_message}")

    # Send the message
    compliance_pass, sender_message_id, receiver_message_id, thread_id, thread_msg_ordering, connection = send_message(sender_company_name,
                                                                                                                       receiver_company_name,
                                                                                                                       sender_division_id,
                                                                                                                       receiver_division_id,
                                                                                                                       clean_message,
                                                                                                                       thread_id if thread_id else "new thread")
    
    if not compliance_pass:
        return None
    
    message_info = {
        "sender_company_name": sender_company_name,
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

def fetch_thread_context(company_name, thread_id, model_type, max_tokens, print_statements=False):
    table_name = format_company_table_name(company_name, "messages")
    messages = supabase.table(table_name).select("*").eq("thread_id", thread_id).order("thread_msg_ordering", desc=True).execute()

    # if messages.error:
    #     print("Error fetching thread context:", messages.error.message)
    #     return None

    accumulated_tokens = 0
    context_messages = []

    for message in reversed(messages.data):
        # Get the proper way to count tokens for a given FM provider
        message_tokens = message['token_counts'].get(model_type, 0)
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

def present_message_with_context(message_info, model_type, model_name, print_statements=False):
    # Default conservative context window length
    default_window_length = 4096
    
    # Query the database for the specific model's context window length
    result = supabase.table("llm_context_windows").select("context_window_length").eq("model_provider", model_type).eq("model_name", model_name).execute()
    
    if result.data:
        max_tokens = result.data[0]['context_window_length']
    else:
        max_tokens = default_window_length
        print(f"Using default max tokens as specific model not found: {max_tokens}")

    thread_context = fetch_thread_context(
        message_info['receiver_company_name'],
        message_info['thread_id'],
        model_type,
        max_tokens=max_tokens
    )

    # if not thread_context:
    #     print("Error: Could not retrieve context for the thread.")
    #     return None

    division_mapping = {
        message_info['sender_division_id']: [message_info['sender_company_name'], message_info['sender_division_tag']],
        message_info['receiver_division_id']: [message_info['receiver_company_name'], message_info['receiver_division_tag']]
    }

    formatted_chain = format_message_chain(thread_context, division_mapping)
    if print_statements:
        print(formatted_chain)
    return formatted_chain


### SIMULATE AGENT THREADS ###

def get_agent_response(agent_name, message):

    agent_response = ""
    # print(f"message being fed into {agent_name}: \n{message}")

    if agent_name == "gfc_tech":
        agent_prompt = f"{agents[agent_name]['system_prompt']}\n {message}"
        # print(f"agent_prompt: {agent_prompt}")

        agent_response = genai.GenerativeModel(default_models["google"]).generate_content(agent_prompt).text

    elif agent_name == "bloomberg_cs":

        agent_prompt = f"{agents[agent_name]['system_prompt']}\n {message}"

        agent_response = client.chat.completions.create(
            messages=[
                { 
                    "role" : "system", 
                    "content" : agents[agent_name]["system_prompt"]
                },
                {
                    "role": "user",
                    "content": message,
                }
            ],
            model=default_models["openai"],
            temperature=0
        ).choices[0].message.content
        # print("after openai response")

    return agent_response

agents = {
    "bloomberg_cs" : {
        "sender_company_name" : "bloomberg",
        "sender_division_id" : 4,
        "model_type" : "openai",
        "model_name" : "GPT-4 Turbo",
        "system_prompt" : "You are a helpful assistant that specializing in working with companies in the finance industry to build custom data integrations with Bloomberg for their data science, risk managment, and trading teams. Your communication style should be very logical and concise, and you are very results-driven. You should produce tangible work products."
    },
    "gfc_tech" : {
        "sender_company_name" : "global financial corp",
        "sender_division_id" : 1,
        "model_type" : "google",
        "model_name" : "Gemini 1.5 Pro",
        "system_prompt" : "You are a helpful assistant that specializing in developing alternative data solutions for investors and traders. Your communication style should be very logical and concise, and you are very results-driven. You should produce tangible work products."
    }  
}

msg_num = 0
sender_idx, receiver_idx = 0, 1
agent_msg_order = ["gfc_tech", "bloomberg_cs"]
thread_num = "new_thread"
end_condition = False
initial_message = f"@{agent_msg_order[receiver_idx]} @{thread_num} Let's design a custom data pipeline that pulls public financial data for a given stock over the past week using the public Bloomberg API then pairs this information with relevant news for the company over the past week using the Perplexity API. I want you to write the first version of code for this pipeline." 
#Please write a python script that uses publicly available APIs to pull in public data for the stock prices of semiconductor companies from the past week and display them in a graph.

while not end_condition:
    print("agent communication intiated...\n")
    if msg_num == 0:
        message_to_send = initial_message
    else:
        message_with_context = present_message_with_context(message_info, agents[agent_msg_order[sender_idx]]["model_type"], agents[agent_msg_order[sender_idx]]["model_name"], print_statements=False)
        message_with_context = f'{message_with_context} \n\n **If you think the original requested task is completed as exactly specified, then simply respond only with "TASK COMPLETE". If you have questions or need clarifications, then please follow-up. Iterate on the task until the request is completed as exactly specified.**'
        # print(f"message_with_context: {message_with_context}")

        agent_response = get_agent_response(agent_msg_order[sender_idx], message_with_context)
        if agent_response == "TASK COMPLETE":
            end_condition = True
    
        thread_str = f"thread_{thread_num}" if thread_num != "new_thread" else thread_num
        message_to_send = f"@{thread_str} @{agent_msg_order[receiver_idx]} {agent_response}"
    
    print(f"Message #{msg_num} from {agent_msg_order[sender_idx]}: \n{message_to_send}\n")
    message_info = auto_send_message(sender_company_name=agents[agent_msg_order[sender_idx]]["sender_company_name"],
                                     sender_division_id=agents[agent_msg_order[sender_idx]]["sender_division_id"],
                                     message=message_to_send)
    
    if not message_info:
        print("Message did not pass compliance check.")
        break
    
    msg_num += 1

    if not sender_idx:
        sender_idx = 1
        receiver_idx = 0
    else:
        sender_idx = 0
        receiver_idx = 1

    if thread_num == "new_thread":
        thread_num = message_info['thread_id']
        print(f"THREAD_NUM: {thread_num}")
