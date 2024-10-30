import fitz
import torch
import os

import numpy as np

from semantic_router.encoders import OpenAIEncoder
from semantic_router.splitters import RollingWindowSplitter

from global_vars import logger, supabase, company_name, openai_client, is_2xx_status_code
from database_interaction import find_connection

### RAG FOR DATA POLICIES ###

encoder = OpenAIEncoder(name="text-embedding-ada-002", openai_api_key=openai_client.api_key)

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
    response = openai_client.embeddings.create(
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
    chunk_result = None
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
            logger.error(f"Error inserting data policy chunk.")
            continue
        
    return chunk_result

def split_store_chunks():
    """
    Fetch PDFs from Supabase Storage, extract text, split into chunks,
    and store chunks along with embeddings in the database.
    """
    # Fetch list of PDFs from storage
    storage_response = supabase.storage.from_('data_policy_documents').list()
    if not storage_response.data:
        logger.error("Error fetching files from storage.")
        return

    files = storage_response.data
    for file_info in files:
        file_name = file_info['name']
        # Download the PDF file
        download_response = supabase.storage.from_('data_policy_documents').download(file_name)
        if not download_response.data:
            logger.error(f"Error downloading file {file_name}.")
            continue
        file_content = download_response.content
        # Save the file temporarily
        temp_file_path = f"/tmp/{file_name}"
        with open(temp_file_path, 'wb') as f:
            f.write(file_content)

        document_id_response = supabase.table("data_policy_doc_infos").select("document_id").eq("file_name", file_name).execute()
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
    document_ids_result = supabase.table("document_connections").select("document_id").eq("connection_id", connection['id']).execute()
    if not document_ids_result.data:
        print("No data policy documents found for this connection.")
        return None
    document_ids = [item['document_id'] for item in document_ids_result.data]

    # Fetch all document_chunk_ids in one query
    document_chunk_ids_result = supabase.table("data_policy_doc_chunks") \
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
