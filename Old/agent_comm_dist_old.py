import os
import re
import fitz
import faiss
import torch
import tiktoken
import requests
import logging

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
# from pyvis.network import Network
# from IPython.display import HTML, display
from vertexai.preview import tokenization
from flask import Flask, request, jsonify # type: ignore

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from semantic_router.splitters import RollingWindowSplitter
from semantic_router.utils.logger import logger
from semantic_router.encoders import OpenAIEncoder

from transformers import LlamaTokenizer
from transformers import AutoTokenizer
# from transformers import T5Tokenizer

'''

next steps:
- make sure that the dockerfile for central database is correct
- make sure the folder structure is all correct
- confirm that yaml and yml are the same thing
- make a dockerfile for enterprise division if needed
- ask what the SUPABASE_KEY and SUPABASE_URL is in the yml file for central database? is this my actual supabase api key or custom?
- should i delete the api keys below? probably just the supabase one but i will likely need the openai and google ones
- see if i need to change the yml file for enterprise division now that we've implemented an endpoint for receiving messages
- start up the docker container for the central database and enterprise division and make sure the data tables are all there
- insert some data into the tables to get going
    - maybe create python file to do this
    - this is what is listed in the tbd
- figure out how to test all this shit
    - we want to start with a super simple set up: two divisions, one connection, one central database
    - this means we will be communicating with three docker deployments

tbd:
- create a separate set_up_network.py file to create a sample network?
    - this actually might not be necessary since the network will actually just be a few different docker containers
- figure out how to import flask lol
    - sort of resolved: the package is installed in the venv but still gets underlined so ignoring the warning
    
'''

### GLOBAL VARIABLES ###

# Connect to Supabase
# url: str = "https://rtzkvrxmbdwyydercpzh.supabase.co"
# key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ0emt2cnhtYmR3eXlkZXJjcHpoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjEwMTY3MDcsImV4cCI6MjAzNjU5MjcwN30.waykwg4OMiDpT0TNU-95dP45oxhhdv8T6rz8yTy0cNo"
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(supabase_url, supabase_key)


# openai_api_key = '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'
# os.environ['OPENAI_API_KEY'] = openai_api_key

client = OpenAI(
    api_key=os.getenv("OPENAI_KEY"),
)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# google_api_key = "AIzaSyATwXkB_Nngdqq1rW5Hna9cwOhJM26YWcA"
# os.environ['GOOGLE_API_KEY'] = google_api_key
genai.configure(api_key=os.getenv["GOOGLE_KEY"])

company_name = os.getenv('COMPANY_NAME')

logger = logging.getLogger(__name__)

### API URLS ###

# updated
CENTRAL_API_BASE_URL = 'http://central-database-url.com'  # Replace with actual URL

# updated
def get_division_api_url(division_id):
    # Query the central database via API to get the division's API URL
    response = requests.get(f"{CENTRAL_API_BASE_URL}/divisions/{division_id}/api_url")
    if response.status_code == 200:
        data = response.json()
        return data.get('api_url')
    else:
        error_message = response.json().get('error', 'Unknown error')
        print(f"Error getting division API URL: {response.status_code} - {error_message}")
        return None


### DATABASE SCHEMA ###

# def register_company(name):
#     data = {"name": name}
#     result = supabase.table("companies").insert(data).execute()
#     return result.data[0]['id']

# updated
def register_company(name):
    data = {"name": name}
    response = requests.post(f"{CENTRAL_API_BASE_URL}/companies", json=data)
    if response.status_code == 201:
        result = response.json()
        return result['id']
    else:
        print(f"Error registering company: {response.status_code} - {response.text}")
        return None


# def register_division(company_id, name, tag):
#     data = {"company_id": company_id, "name": name, "tag": tag}
#     print(data)
#     result = supabase.table("divisions").insert(data).execute()
#     return result.data[0]['id']

# updated
def register_division(company_id, name, tag):
    data = {"company_id": company_id, "name": name, "tag": tag}
    response = requests.post(f"{CENTRAL_API_BASE_URL}/divisions", json=data)
    if response.status_code == 201:
        result = response.json()
        return result['id']
    else:
        print(f"Error registering division: {response.status_code} - {response.text}")
        return None


# def add_connection(source_division_id, target_division_id, daily_messages_count=0):
#     data = {"source_division_id": source_division_id, "target_division_id": target_division_id, "daily_messages_count": daily_messages_count}
#     result = supabase.table("connections").insert(data).execute()
#     return result.data[0]['id']

# updated
def add_connection(source_division_id, target_division_id, daily_messages_count=0):
    data = {
        "source_division_id": source_division_id,
        "target_division_id": target_division_id,
        "daily_messages_count": daily_messages_count
    }
    response = requests.post(f"{CENTRAL_API_BASE_URL}/connections", json=data)
    if response.status_code == 201:
        result = response.json()
        return result['id']
    else:
        print(f"Error adding connection: {response.status_code} - {response.text}")
        return None

# def find_connection(sender_division_id, receiver_division_id):
#     # Query for the connection in both possible directions
#     query1 = supabase.table("connections").select("*").eq("source_division_id", sender_division_id).eq("target_division_id", receiver_division_id).execute()
#     query2 = supabase.table("connections").select("*").eq("source_division_id", receiver_division_id).eq("target_division_id", sender_division_id).execute()

#     if query1.data:
#         return query1.data[0]  # Return the first result if available
#     elif query2.data:
#         return query2.data[0]  # Return the first result from the second query if available
#     else:
#         return None  # No connection found

# updated
def find_connection(sender_division_id, receiver_division_id):
    params = {
        'division_id_1': sender_division_id,
        'division_id_2': receiver_division_id
    }
    response = requests.get(f"{CENTRAL_API_BASE_URL}/connections", params=params)
    if response.status_code == 200:
        connections = response.json()
        if connections:
            return connections[0]
        else:
            return None  # No connection found
    else:
        error_message = response.json().get('error', 'Unknown error')
        print(f"Error querying central database: {response.status_code} - {error_message}")
        return None  # Early exit on error

# def insert_data_policy(sender_division_id, receiver_division_id, confidentiality, data_type, explanation):
#     # Look up the connection ID using sender and receiver division IDs
#     print(sender_division_id, receiver_division_id)
#     connection = find_connection(sender_division_id, receiver_division_id)

#     if not connection:
#         return "No valid connection found between the specified divisions."

#     connection_id = connection['id']
#     print(f"Data policy for connection #{connection_id}: {explanation}")

#     # Determine which company owns the sending division to decide where to store the policy
#     company_info = supabase.table("divisions").select("company_id").eq("id", sender_division_id).execute()
#     if not company_info.data:
#         return "Sending division not found."
#     company_id = company_info.data[0]['company_id']

#     # Retrieve the company name based on company_id for table naming
#     company = supabase.table("companies").select("name").eq("id", company_id).execute()
#     if not company.data:
#         return "Company not found."
#     company_name = company.data[0]['name']

#     # Format the table name from the company name
#     table_name = format_company_table_name(company_name, "data_policies")

#     # Insert data into the company-specific data policies table
#     data = {
#         "connection_id": connection_id,
#         "confidentiality": confidentiality,
#         "data_type": data_type,
#         "natural_language_explanation": explanation
#     }
#     result = supabase.table(table_name).insert(data).execute()
#     return result.data

# updated
def insert_data_policy(sender_division_id, receiver_division_id, confidentiality, data_type, explanation):
    # Look up the connection using API call to central database
    connection = find_connection(sender_division_id, receiver_division_id)
    if not connection:
        return "No valid connection found between the specified divisions."
    
    connection_id = connection['id']

    # Insert data into the division's own 'custom_data_policies' table
    data = {
        "connection_id": connection_id,
        "confidentiality": confidentiality,
        "data_type": data_type,
        "natural_language_explanation": explanation
    }
    result = supabase.table("custom_data_policies").insert(data).execute()
    return result.data

# this function is useless because the data policy table is now local
# def get_data_policies(company_id, division_id=None, receiving_division_id=None):
#     # Retrieve the company name based on company_id using the central API
#     response = requests.get(f"{CENTRAL_API_BASE_URL}/companies/{company_id}")
#     if response.status_code == 200:
#         company_info = response.json()
#     else:
#         return "Company not found"

#     company_name = company_info['name']
#     table_name = format_company_table_name(company_name, "data_policies")

#     # Build the query based on the level of specificity required
#     query = supabase.table(table_name).select("*")

#     if division_id:
#         # Fetch connections for the specific division, either as source or target
#         connections = supabase.table("connections").select("id").or_(
#             f"source_division_id.eq.{division_id},target_division_id.eq.{division_id}"
#         ).execute()
#         if not connections.data:
#             return "No connections found for the specified division"
#         connection_ids = [conn['id'] for conn in connections.data]
#         query = query.in_("connection_id", connection_ids)
#     elif receiving_division_id:
#         # Fetch policies for all company divisions connected to a specified receiving division
#         connections = supabase.table("connections").select("id").or_(
#             f"source_division_id.eq.{receiving_division_id},target_division_id.eq.{receiving_division_id}"
#         ).execute()
#         if not connections.data:
#             return "No connections found for the specified receiving division"
#         connection_ids = [conn['id'] for conn in connections.data]
#         query = query.in_("connection_id", connection_ids)

#     # Execute the query and return the results
#     result = query.execute()
#     return result.data if result.data else "No policies found"

# another useless function
def format_company_table_name(company_name, table_type=None):
    # Remove commas and periods, replace spaces with underscores, convert to lowercase
    formatted_name = company_name.replace(",", "").replace(".", "").replace(" ", "_").lower()
    if table_type:
        table_type_name = f"_{table_type}"
    else:
        table_type_name = ""
    return f"{formatted_name}{table_type_name}"

# unused
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
    chunk = f"--- The following passage is an extract from a data policy document for {company_name} ---\n\n {content}"
    if print_chunk:
        print(chunk)
    return chunk

def store_policy_chunks(text_chunks, document_id):
    for chunk_order, chunk in enumerate(text_chunks):
        embedding = generate_embeddings(chunk)
        # Insert chunk into the data_policy_doc_chunks table
        chunk_result = supabase.table("data_policy_doc_chunks").insert({
            "content": chunk,
            "embedding": embedding,
            "document_id": document_id,
            "chunk_order": chunk_order
        }).execute()
        if not chunk_result.data:
            print(f"Error inserting data policy chunk.")
            continue
        
    return chunk_result

        # if result.error:
        #     print("Error inserting data chunk:", result.error.message)
        #     continue

        # Retrieve the document_chunk_id from the insertion result
        # document_chunk_id = result.data[0]['id']
        
        # TODO: i dont think this is necessary anymore
        # Insert connections for this document chunk
        # for connection_id in applicable_connections:
        #     connection_result = supabase.table("document_connections").insert({
        #         "document_chunk_id": document_chunk_id,
        #         "connection_id": connection_id
        #     }).execute()

            # if connection_result.error:
            #     print("Error linking document chunk to connection:", connection_result.error.message)


# TODO: we should store these pdfs in supabase storage (right now we are just storying file paths to where they are stored locally)
# TODO: we should also add a function for registering a new data policy doc (i.e., adding a new pdf to the storage bucket)
# TODO: enterprises will be dropping in their data policy PDFs using the UI that we eventually build out, so adding a doc to storage will actually be a javascript function but we will need to write in this in a new file because agent_comm_dist.py is a python file 
# TODO: let's link these docs to the data_policy_docs data table via the document_num column (not sure if we can assign a document_num to each pdf in storage)
# data_policy_doc_infos = {
#     "Bloomberg" : [
#         {"location": "/Data Policies/Bloomberg Data Policies.pdf",
#          "applicable_connections" : [1, 4]
#         }
#     ],
#     "Global Financial Corp" : [
#         {"location": "/Data Policies/Global Financial Corp Data Policies.pdf",
#          "applicable_connections" : [1, 2]
#         }
#     ]
# }



# TODO: this function is for initially storing the chunks from the data_policy_doc_infos
# TODO: do we really need the data type param? if we do, how we can determine the data type the policy is applicable to? (this is lower priority)
# TODO: we will need to query the data policy doc pdfs in storage, and because it's local, we don't need a company_name
# TODO: since we won't have the company name because this data policy doc pdfs are stored locally, we should just have the company name as a .env variable (we still need the company name for build_text_chunk())
# def split_store_chunks(data_policy_doc_infos):
#     for company_name, infos in data_policy_doc_infos.items():
#         for doc_num, info in enumerate(infos):
#             location = info['location']
#             # applicable_connections = info['applicable_connections']
#             text = extract_text_from_pdf(location)
#             splits = splitter([text])
#             policy_chunks = [build_text_chunk(company_name, split.content, True) for split in splits]
#             store_policy_chunks(policy_chunks, doc_num)

# updated
def split_store_chunks():
    """
    Fetch PDFs from Supabase Storage, extract text, split into chunks,
    and store chunks along with embeddings in the database.
    """
    # Fetch list of PDFs from storage
    response = supabase.storage.from_('data_policy_documents').list()
    if response.status_code != 200:
        print("Error fetching files from storage.")
        return

    files = response.data
    for file_info in files:
        file_name = file_info['name']
        # Download the PDF file
        download_response = supabase.storage.from_('data_policy_documents').download(file_name)
        if download_response.status_code != 200:
            print(f"Error downloading file {file_name}.")
            continue
        file_content = download_response.content
        # Save the file temporarily
        temp_file_path = f"/tmp/{file_name}"
        with open(temp_file_path, 'wb') as f:
            f.write(file_content)

        document_id_response = supabase.from_("data_policy_doc_infos").select("document_id").eq("file_name", file_name).execute()
        if document_id_response.data:
            document_id = document_id_response.data[0]['document_id']
        else:
            print(f"No document_id found for file_name {file_name}.")
            continue

        # Extract text and process
        text = extract_text_from_pdf(temp_file_path)
        splits = splitter([text])
        policy_chunks = [build_text_chunk(company_name, split.content, True) for split in splits]
        store_policy_chunks(policy_chunks, document_id)

        # Remove the temporary file
        os.remove(temp_file_path)


# def retrieve_relevant_document_chunks(query, company_name, connection, threshold=0.7, top_k=5, sender_division_id=None, receiver_division_id=None):
#     # For testing in isolation
#     if not connection:
#         if sender_division_id and receiver_division_id:
#             connection = find_connection(sender_division_id, receiver_division_id)
#         else:
#             return "No valid connection found or provided."

#     # Generate query embedding
#     query_embedding = generate_embeddings(query)

#     # Prepare the embedding for database compatibility, typically converting a tensor to a list of floats
#     if isinstance(query_embedding, np.ndarray):
#         query_embedding = query_embedding.tolist()
#     elif isinstance(query_embedding, torch.Tensor):
#         query_embedding = query_embedding.numpy().tolist()

#     # Get the IDs of document chunks applicable to this connection
#     # TODO: i don't think we need this line because document_connections is a local data table and therefore does not need the company name prepended 
#     document_connection_table = format_company_table_name(company_name, "document_connections")
#     # TODO: i think this should just be  supabase.from_("document_connections").select("document_chunk_id").eq("connection_id", connection['id']).execute()
#     document_chunk_ids = supabase.from_(document_connection_table).select("document_chunk_id").eq("connection_id", connection['id']).execute()
#     # if document_chunk_ids.error:
#     #     print("Error fetching document chunk IDs:", document_chunk_ids.error.message)
#     #     return None
#     document_chunk_ids = [item['document_chunk_id'] for item in document_chunk_ids.data]

#     # TODO: same here -- data_policy_docs is a local datatable
#     # Now invoke the RPC function to get relevant document chunks based on embeddings
#     table_name = format_company_table_name(company_name, "data_policy_docs")
#     result = supabase.rpc('match_document_chunks', {
#         'table_name': table_name,
#         'document_chunk_ids': document_chunk_ids,
#         'query_embedding': query_embedding,
#         'match_threshold': threshold,
#         'match_count': top_k
#     }).execute()

#     # if result.error:
#     #     print("Error during retrieval:", result.error.message)
#     #     return None

#     return result.data

# updated
def retrieve_relevant_document_chunks(query, connection, threshold=0.7, top_k=5, sender_division_id=None, receiver_division_id=None):
    # For testing in isolation
    if not connection:
        if sender_division_id and receiver_division_id:
            connection = find_connection(sender_division_id, receiver_division_id)
        else:
            return "No valid connection found or provided."

    # Generate query embedding
    query_embedding = generate_embeddings(query)

    # Prepare the embedding for database compatibility
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()
    elif isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.numpy().tolist()

    # Get the IDs of document chunks applicable to this connection
    document_ids_result = supabase.from_("document_connections").select("document_id").eq("connection_id", connection['id']).execute()
    if not document_ids_result.data:
        print("No data policy documents found for this connection.")
        return None
    document_ids = [item['document_id'] for item in document_ids_result.data]

    # Fetch all document_chunk_ids in one query
    document_chunk_ids_result = supabase.from_("data_policy_doc_chunks") \
        .select("id") \
        .in_("document_id", document_ids) \
        .execute()
    if not document_chunk_ids_result.data:
        print("No document chunks found for these documents.")
        return None
    document_chunk_ids = [item['id'] for item in document_chunk_ids_result.data]

    # Now invoke the RPC function to get relevant document chunks based on embeddings
    result = supabase.rpc('match_document_chunks', {
        'table_name': 'data_policy_doc_chunks',
        'document_chunk_ids': document_chunk_ids,
        'query_embedding': query_embedding,
        'match_count': top_k
    }).execute()

    return result.data


### UPLOAD AND REGISTER DATA POLICY DOCS ###

# updated
response = supabase.storage.create_bucket('data_policy_documents')

def upload_pdf_to_storage(file_path, file_name):
    """
    Uploads a PDF file to Supabase Storage.

    Args:
        file_path (str): The local path to the PDF file.
        file_name (str): The name under which to store the file in the bucket.

    Returns:
        dict: Information about the uploaded file, or None if upload failed.
    """
    file_name = os.path.basename(file_name)
    with open(file_path, 'rb') as f:
        file_content = f.read()
    response = supabase.storage.from_('data_policy_documents').upload(file_name, file_content)
    if response.status_code == 200:
        return response.data  # Contains information about the uploaded file
    else:
        print(f"Error uploading file: {response.status_code} - {response.text}")
        return None

# updated
def register_data_policy_doc(file_name):
    """
    Registers a new data policy document in the database.

    Args:
        file_name (str): The name of the file in the storage bucket.
        applicable_connections (list): List of connection IDs this policy applies to.

    Returns:
        int: The assigned document_num, or None if registration failed.
    """
    # Insert a new record into the data_policy_docs table to get a document_num
    data = {
        "file_name": file_name,
    }
    result = supabase.table("data_policy_doc_infos").insert(data).execute()
    if result.data:
        document_id = result.data[0]['document_id']
        return document_id
    else:
        print(f"Error registering data policy document with file_name {file_name}.")
        return None

def register_data_policy_doc_connections(document_id, applicable_connections):
    # Insert connections for this document chunk
    for connection_id in applicable_connections:
        doc_connection_result = supabase.table("document_connections").insert({
            "document_id": document_id,
            "connection_id": connection_id
        }).execute()
        if not doc_connection_result.data:
            print(f"Error registering data policy document connection. (document_id {document_id}, connection_id {connection_id})")
            return None
    return doc_connection_result
    

def upload_register_data_policy_doc(file_path, file_name, applicable_connections):
    try:
        # Begin transaction
        supabase.rpc('begin').execute()

        upload_result = upload_pdf_to_storage(file_path, file_name)
        if not upload_result:
            raise Exception("Failed to upload PDF to storage.")

        document_id = register_data_policy_doc(file_name)
        if not document_id:
            raise Exception("Failed to register data policy document.")

        doc_connection_result = register_data_policy_doc_connections(document_id, applicable_connections)
        if not doc_connection_result:
            raise Exception("Failed to register data policy document connections.")

        # Commit transaction
        supabase.rpc('commit').execute()
        return "Successfully stored, registered, and connected data policy document."
    except Exception as e:
        # Rollback transaction
        supabase.rpc('rollback').execute()
        logger.error(f"Error in upload_register_data_policy_doc: {e}")
        return None



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

# TODO: maybe add a column for the official name of the model as it is stored in the default models dict? not sure if this is really needed though because this is really just a look up table and does not interact with the actual counting of tokens
# TODO: this is unused currently -- maybe create a new .py file with function that alter the central database for the central admin. this is probably where the set_up_network() function(s) should be.
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

# def log_message(sender_company_name, receiver_company_name, connection, sender_division_id, receiver_division_id, message, thread_id="new thread", print_statements=False):
#     sender_table_name = format_company_table_name(sender_company_name, "messages")
#     receiver_table_name = format_company_table_name(receiver_company_name, "messages")
#     threads_table_name = get_threads_table_name(connection['id'])

#     # Calculate token counts for all model providers 
#     token_counts = {}
#     for provider, default_model in default_models.items():
#         count_function = tokenizers[provider]
#         token_counts[provider] = count_function(message, default_model)
#     if print_statements:
#         print(f"token_counts in log_message(): {token_counts}")

#     if isinstance(thread_id, str):
#         if thread_id.lower() == "new thread":
#             thread_entry = {
#                 "connection_id": connection['id'],
#                 "last_message_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "source_division_cost": 0.0001 if sender_division_id == connection['source_division_id'] else 0,
#                 "target_division_cost": 0.0001 if sender_division_id == connection['target_division_id'] else 0,
#                 "messages_count": 1
#             }
#             thread_result = supabase.table(threads_table_name).insert(thread_entry).execute()
#             thread_id = thread_result.data[0]['thread_id']
#             thread_msg_ordering = 1
#         else:
#             print("Invalid thread_id or new thread.")
#             return None, None, None, None
#     else:
#         # Existing thread handling
#         thread = supabase.table(threads_table_name).select("*").eq("thread_id", thread_id).execute().data[0]
#         update_data = {
#             "messages_count": thread['messages_count'] + 1,
#             "last_message_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "source_division_cost": thread['source_division_cost'] + (0.0001 if sender_division_id == connection['source_division_id'] else 0),
#             "target_division_cost": thread['target_division_cost'] + (0.0001 if sender_division_id == connection['target_division_id'] else 0)
#         }
#         supabase.table(threads_table_name).update(update_data).eq("thread_id", thread_id).execute()
#         thread_msg_ordering = thread['messages_count'] + 1

#     message_entry = {
#         "connection_id": connection['id'],
#         "sender_division_id": sender_division_id,
#         "receiver_division_id": receiver_division_id,
#         "message_content": message,
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "status": "sent",
#         "thread_id": thread_id,
#         "thread_msg_ordering": thread_msg_ordering,
#         "token_counts": token_counts
#     }
#     if print_statements:
#         print(f"message_entry in log_message(): {message_entry}")
#     sender_result = supabase.table(sender_table_name).insert(message_entry).execute()
#     receiver_message_entry = message_entry.copy()
#     receiver_message_entry['status'] = "received"
#     receiver_result = supabase.table(receiver_table_name).insert(receiver_message_entry).execute()

#     return sender_result.data[0]['message_id'], receiver_result.data[0]['message_id'], thread_id, thread_msg_ordering

# updated
def log_message(sender_division_id, receiver_division_id, connection, message, thread_id="new thread", print_statements=False):
    # Calculate token counts
    token_counts = {}
    for provider, default_model in default_models.items():
        count_function = tokenizers[provider]
        token_counts[provider] = count_function(message, default_model)
    if print_statements:
        print(f"token_counts in log_message(): {token_counts}")

    if thread_id == "new thread":
        # Create a new thread in the central database via API call
        thread_entry = {
            "connection_id": connection['id'],
            "last_message_timestamp": datetime.now().isoformat(),
            "source_division_cost": 0.0001 if sender_division_id == connection['source_division_id'] else 0,
            "target_division_cost": 0.0001 if sender_division_id == connection['target_division_id'] else 0,
            "messages_count": 1
        }
        create_thread_response = requests.post(f"{CENTRAL_API_BASE_URL}/threads", json=thread_entry)
        if create_thread_response.status_code == 201:
            thread_data = create_thread_response.json()
            thread_id = thread_data['thread_id']
            thread_msg_ordering = 1
        else:
            print("Error creating thread in central database.")
            return None, None, None, None
    else:
        # Update existing thread in the central database via API call
        response = requests.get(f"{CENTRAL_API_BASE_URL}/threads/{thread_id}")
        if response.status_code == 200:
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
            if update_thread_response.status_code != 200:
                print("Error updating thread in central database.")
                return None, None, None, None
            thread_msg_ordering = thread_data['messages_count'] + 1
        else:
            print("Error accessing thread in central database.")
            return None, None, None, None

    message_entry = {
        "connection_id": connection['id'],
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

    # Prepare the data to send
    data_to_send = {
        "message_entry": message_entry
    }

    response = requests.post(f"{receiver_api_url}/api/messages/receive", json=data_to_send)
    if response.status_code == 200:
        receiver_response = response.json()
        receiver_message_id = receiver_response.get('message_id')
    else:
        print(f"Error sending message to receiver division: {response.status_code}")
        return None, None, None, None

    return sender_result.data[0]['message_id'], receiver_message_id, thread_id, thread_msg_ordering



# def send_message(sender_company_name, receiver_company_name, sender_division_id, receiver_division_id, message, thread_id="new thread", print_statements=False):

#     # Retrieve company-specific connection using sender and receiver names
#     connection = find_connection(sender_division_id, receiver_division_id)
#     if not connection:
#         return "No valid connection found."

#     data_policies_table_name = format_company_table_name(sender_company_name, "data_policies")

#     # Retrieve data policies for the connection
#     policies = supabase.table(data_policies_table_name).select("*").eq("connection_id", connection['id']).execute()

#     if policies.data:
#         for policy in policies.data:
#             compliant_bool, compliance_response = check_policy_compliance(message, policy)
#             if not compliant_bool:
#                 print(f"Message violates policy: \n{policy['natural_language_explanation']} \n\nCompliance response: \n{compliance_response}")
#                 return False, None, None, None, None, None
#     else:
#         print("No data policies found for this connection. Proceeding with message sending.")

#     relevant_chunks = retrieve_relevant_document_chunks(message, sender_company_name, connection, top_k=2)
#     if relevant_chunks:
#         for chunk in relevant_chunks:
#             compliant_bool, compliance_response = check_policy_compliance(message, chunk['content'], from_doc=True)
#             if not compliant_bool:
#                 print(f"Message violates a policy from this excerpt: \n{chunk['content']} \n\nCompliance response: \n{compliance_response}")
#                 return False, None, None, None, None, None
#     else:
#         print("No relevant document chunks found. Proceeding with message sending.")

#     # Log and update the message thread
#     sender_message_id, receiver_message_id, thread_id, thread_msg_ordering = log_message(
#         sender_company_name, receiver_company_name, connection, sender_division_id, receiver_division_id, message, thread_id
#     )

#     if print_statements:
#         print(f"Message sent successfully. Sender message ID: {sender_message_id}, Receiver message ID: {receiver_message_id}, Thread ID: {thread_id}, Message Ordering: {thread_msg_ordering}")

#     return True, sender_message_id, receiver_message_id, thread_id, thread_msg_ordering, connection

# updated
def send_message(sender_division_id, receiver_division_id, message, thread_id="new thread", print_statements=False):
    # Find connection via API call
    connection = find_connection(sender_division_id, receiver_division_id)
    if not connection:
        print("No valid connection found.")
        return False, None, None, None, None, None

    # Retrieve data policies from local database
    policies = supabase.table("custom_data_policies").select("*").eq("connection_id", connection['id']).execute()

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
    relevant_chunks = retrieve_relevant_document_chunks(message, connection, top_k=2)
    if relevant_chunks:
        for chunk in relevant_chunks:
            compliant_bool, compliance_response = check_policy_compliance(message, chunk['content'], from_doc=True)
            if not compliant_bool:
                print(f"Message violates a policy from this excerpt: \n{chunk['content']} \n\nCompliance response: \n{compliance_response}")
                return False, None, None, None, None, None
    else:
        print("No relevant document chunks found. Proceeding with message sending.")

    # Log message
    sender_message_id, receiver_message_id, thread_id, thread_msg_ordering = log_message(
        sender_division_id, receiver_division_id, connection, message, thread_id
    )

    if print_statements:
        print(f"Message sent successfully. Sender message ID: {sender_message_id}, Receiver message ID: {receiver_message_id}, Thread ID: {thread_id}, Message Ordering: {thread_msg_ordering}")

    return True, sender_message_id, receiver_message_id, thread_id, thread_msg_ordering, connection


# def auto_send_message(sender_company_name, sender_division_id, message, print_statements=False):
#     # Regex patterns to find tags
#     division_tag_pattern = re.compile(r"@(?!thread_|new_thread)([a-zA-Z0-9_]+)")
#     thread_tag_pattern = re.compile(r"@thread_(\d+)")
#     new_thread_tag_pattern = re.compile(r"@new_thread")

#     # Extract tags from the message
#     division_tags = division_tag_pattern.findall(message)
#     thread_tags = thread_tag_pattern.findall(message)
#     new_thread_tags = new_thread_tag_pattern.findall(message)

#     # Check for explicit new thread tag
#     if new_thread_tags:
#         thread_id = "new thread"
#     elif thread_tags:
#         if len(thread_tags) > 1:
#             print("Error: Multiple thread tags detected. Please specify only one.")
#             return
#         thread_id = int(thread_tags[0])
#     else:
#         thread_id = None

#     if len(division_tags) > 1:
#         print("Error: Multiple division tags detected. Please specify only one.")
#         return None

#     if not division_tags and not thread_id:
#         print("No appropriate tags found. No message sent.")
#         return None

#     # Find the receiving division and its corresponding company if a division tag is present
#     if division_tags:
#         receiver_division_tag = division_tags[0]
#         division_info = supabase.table("divisions").select("id", "company_id").eq("tag", receiver_division_tag).execute()
#         if not division_info.data:
#             print("Error: No division found with the specified tag.")
#             return None
#         receiver_division_id = division_info.data[0]['id']
#         receiver_company_info = supabase.table("companies").select("name").eq("id", division_info.data[0]['company_id']).execute()
#         receiver_company_name = receiver_company_info.data[0]['name']
#     else:
#         print("Division tag is required to send a message.")
#         return None

#     # Check for a valid connection
#     connection = find_connection(sender_division_id, receiver_division_id)
#     if not connection:
#         print("No valid connection found between specified divisions.")
#         return None

#     sender_division_tag = supabase.table("divisions").select("tag").eq("id", sender_division_id).execute().data[0]['tag']

#     # Validate the thread if specified
#     if thread_id and isinstance(thread_id, int):  # Ensure thread_id is an integer for existing threads
#         threads_table_name = get_threads_table_name(connection['id'])
#         thread_info = supabase.table(threads_table_name).select("*").eq("thread_id", thread_id).execute()
#         if not thread_info.data:
#             print(f"No thread found with ID {thread_id} in the specified connection.")
#             return None

#     # Remove tags from the message before sending
#     clean_message = re.sub(r"@[\w_]+", "", message).strip()

#     if print_statements:
#         print(f"Sender: {sender_company_name} (Division ID: {sender_division_id})")
#         print(f"Receiver: {receiver_company_name} (Division ID: {receiver_division_id})")
#         print(f"Thread ID: {thread_id if thread_id else 'new thread'}")
#         print(f"Message: {clean_message}")

#     # Send the message
#     compliance_pass, sender_message_id, receiver_message_id, thread_id, thread_msg_ordering, connection = send_message(sender_company_name,
#                                                                                                                        receiver_company_name,
#                                                                                                                        sender_division_id,
#                                                                                                                        receiver_division_id,
#                                                                                                                        clean_message,
#                                                                                                                        thread_id if thread_id else "new thread")
    
#     if not compliance_pass:
#         return None
    
    # message_info = {
    #     "sender_company_name": sender_company_name,
    #     "receiver_company_name": receiver_company_name,
    #     "sender_division_id": sender_division_id,
    #     "receiver_division_id": receiver_division_id,
    #     "sender_division_tag": sender_division_tag,
    #     "receiver_division_tag": receiver_division_tag,
    #     "message": clean_message,
    #     "thread_id": thread_id,
    #     "thread_msg_ordering": thread_msg_ordering,
    #     "connection_id": connection['id']
    # }

#     return message_info

# updated
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
            print("Error: Multiple thread tags detected. Please specify only one.")
            return None
        thread_id = int(thread_tags[0])
    else:
        thread_id = None

    if len(division_tags) != 1:
        print("Error: Please specify exactly one division tag.")
        return None

    # Get receiver division ID from the tag
    receiver_division_tag = division_tags[0]
    # Query central database via API to get receiver division details
    response = requests.get(f"{CENTRAL_API_BASE_URL}/divisions/tag/{receiver_division_tag}")
    if response.status_code == 200:
        division_info = response.json()
        receiver_division_id = division_info['id']
        receiver_company_id = division_info['company_id']
        receiver_division_tag = division_info['tag']

        # Get receiver company name
        response_company = requests.get(f"{CENTRAL_API_BASE_URL}/companies/{receiver_company_id}")
        if response_company.status_code == 200:
            receiver_company_info = response_company.json()
            receiver_company_name = receiver_company_info['name']
        else:
            print("Error retrieving receiver company information.")
            return None
    else:
        print("Error: No division found with the specified tag.")
        return None

    # Get sender division tag and company name
    # Query central database via API to get sender division info
    response_division = requests.get(f"{CENTRAL_API_BASE_URL}/divisions/{sender_division_id}")
    if response_division.status_code == 200:
        division_info = response_division.json()
        sender_division_tag = division_info['tag']
        sender_company_id = division_info['company_id']

        # Get sender company name
        # TODO: we may just want to pass in sender_company_name to the function like we were doing before
        response_company = requests.get(f"{CENTRAL_API_BASE_URL}/companies/{sender_company_id}")
        if response_company.status_code == 200:
            sender_company_info = response_company.json()
            sender_company_name = sender_company_info['name']
        else:
            print("Error retrieving sender company information.")
            return None
    else:
        print("Error retrieving sender division information.")
        return None

    # Clean message by removing tags
    clean_message = re.sub(r"@[\w_]+", "", message).strip()

    if print_statements:
        print(f"Sender Company: {sender_company_name} (Division ID: {sender_division_id}, Tag: {sender_division_tag})")
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

# def present_message_with_context(message_info, model_type, model_name, print_statements=False):
#     # Default conservative context window length
#     default_window_length = 4096
    
#     # Query the database for the specific model's context window length
#     result = supabase.table("llm_context_windows").select("context_window_length").eq("model_provider", model_type).eq("model_name", model_name).execute()
    
#     if result.data:
#         max_tokens = result.data[0]['context_window_length']
#     else:
#         max_tokens = default_window_length
#         print(f"Using default max tokens as specific model not found: {max_tokens}")

#     thread_context = fetch_thread_context(
#         message_info['receiver_company_name'],
#         message_info['thread_id'],
#         model_type,
#         max_tokens=max_tokens
#     )

#     # if not thread_context:
#     #     print("Error: Could not retrieve context for the thread.")
#     #     return None

    # division_mapping = {
    #     message_info['sender_division_id']: [message_info['sender_company_name'], message_info['sender_division_tag']],
    #     message_info['receiver_division_id']: [message_info['receiver_company_name'], message_info['receiver_division_tag']]
    # }

#     formatted_chain = format_message_chain(thread_context, division_mapping)
#     if print_statements:
#         print(formatted_chain)
#     return formatted_chain

# updated
def present_message_with_context(message_info, model_type, model_name, print_statements=False):

    # Get context window length from central database via API call
    params = {
        'model_provider': model_type,
        'model_name': model_name
    }
    response = requests.get(f"{CENTRAL_API_BASE_URL}/llm_context_windows", params=params)
    if response.status_code == 200:
        data = response.json()
        if data:
            max_tokens = data['context_window_length']
        else:
            max_tokens = 4096  # Default value if not found
            print("Model not found in central database. Using default max tokens.")
    else:
        max_tokens = 4096  # Default value if API call fails
        print(f"Error retrieving context window length: {response.status_code}. Using default max tokens.")

    # Fetch thread context from the division's own database
    thread_context = fetch_thread_context(
        message_info['thread_id'],
        model_type,
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


### SIMULATE AGENT THREADS ###

def get_agent_response(agent_name, message):

    agent_response = ""
    # print(f"message being fed into {agent_name}: \n{message}")

    if agent_name == "gfc_tech":
        agent_prompt = f"{agents[agent_name]['system_prompt']}\n {message}"
        # print(f"agent_prompt: {agent_prompt}")

        agent_response = genai.GenerativeModel(default_models["google"]).generate_content(agent_prompt).text

    elif agent_name == "bloomberg_cs":

        # agent_prompt = f"{agents[agent_name]['system_prompt']}\n {message}"

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


### RECEIVE MESSAGES ###

app = Flask(__name__)

# updated
@app.route('/api/messages/receive', methods=['POST'])
def receive_message():
    try:
        data = request.get_json()
        message_entry = data.get('message_entry')
        if not message_entry:
            return jsonify({"error": "Invalid data"}), 400   
        required_fields = ['connection_id', 'sender_division_id', 'receiver_division_id', 'message_content', 
                        'timestamp', 'status', 'thread_id', 'thread_msg_ordering', 'token_counts']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"'{field}' is required"}), 400

        # Insert the message with status 'received' into local database
        message_entry['status'] = 'received'
        result = supabase.table("messages").insert(message_entry).execute()

        return jsonify({"message_id": result.data[0]['message_id']}), 200
    except Exception as e:
        app.logger.error(f"Error receiving message: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


### SEND AND PRESENT MESSAGES ###

@app.route('/api/auto_send_message', methods=['POST'])
def api_auto_send_message():
    try:
        data = request.get_json()
        sender_division_id = data.get('sender_division_id')
        message = data.get('message')
        if not sender_division_id or not message:
            return jsonify({"error": "Invalid data"}), 400
        # Call the function and return the result
        message_info = auto_send_message(sender_division_id, message)
        return jsonify(message_info)
    except Exception as e:
        app.logger.error(f"Error auto sending message: {e}")
        return jsonify({"error": "Internal Server Error"}), 500        

@app.route('/api/present_message_with_context', methods=['POST'])
def api_present_message_with_context():
    data = request.get_json()
    message_info = data.get('message_info')
    model_type = data.get('model_type')
    model_name = data.get('model_name')
    if not message_info or not model_type or not model_name:
        return jsonify({"error": "Invalid data"}), 400
    context = present_message_with_context(message_info, model_type, model_name)
    return jsonify({'context': context})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
