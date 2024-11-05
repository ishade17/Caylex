import os

from global_vars import logger, supabase, is_2xx_status_code


### UPLOAD AND REGISTER DATA POLICY DOCS ###

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
    if not response.data:
        logger.error(f"Error uploading file: {response.status_code} - {response.text}")
        return None
    return response.data  # Contains information about the uploaded file

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
        logger.error(f"Error registering data policy document with file_name {file_name}.")
        return None

def register_data_policy_doc_connections(document_id, applicable_connections):
    # Insert connections for this document chunk
    for connection_id in applicable_connections:
        doc_connection_result = supabase.table("document_connections").insert({
            "document_id": document_id,
            "connection_id": connection_id
        }).execute()
        if not doc_connection_result.data:
            logger.error(f"Error registering data policy document connection. (document_id {document_id}, connection_id {connection_id})")
            return None
    return doc_connection_result
    
def upload_register_data_policy_doc(file_path, file_name, applicable_connections):
    try:
        upload_result = upload_pdf_to_storage(file_path, file_name)
        if not upload_result:
            raise Exception("Failed to upload PDF to storage.")

        document_id = register_data_policy_doc(file_name)
        if not document_id:
            # Cleanup: Delete the uploaded file since registration failed
            supabase.storage.from_('data_policy_documents').remove([file_name])
            raise Exception("Failed to register data policy document.")

        doc_connection_result = register_data_policy_doc_connections(document_id, applicable_connections)
        if not doc_connection_result:
            # Cleanup: Delete the document record and uploaded file
            supabase.table("data_policy_doc_infos").delete().eq("document_id", document_id).execute()
            supabase.storage.from_('data_policy_documents').remove([file_name])
            raise Exception("Failed to register data policy document connections.")

        return "Successfully stored, registered, and connected data policy document."

    except Exception as e:
        logger.exception(f"Error in upload_register_data_policy_doc: {e}")
        return None

