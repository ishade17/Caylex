import requests
import secrets
import hashlib

from global_vars import supabase, logger, CENTRAL_API_BASE_URL, CENTRAL_API_KEY, is_2xx_status_code

### DATABASE SCHEMA ###

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {CENTRAL_API_KEY}",
}   

def register_company(name):
    data = {"name": name}
    # TODO: figure out if we need only the apikey header or the authorization or both
    response = requests.post(f"{CENTRAL_API_BASE_URL}/companies", json=data, headers=headers)
    if is_2xx_status_code(response.status_code):
        result = response.json()
        return result['id']
    else:
        logger.error(f"Error registering company: {response.status_code} - {response.text}")
        return None

def register_division(company_id, name, tag):
    data = {"company_id": company_id, "name": name, "tag": tag}
    response = requests.post(f"{CENTRAL_API_BASE_URL}/divisions", json=data, headers=headers)
    if is_2xx_status_code(response.status_code):
        result = response.json()
        return result['id']
    else:
        logger.error(f"Error registering division: {response.status_code} - {response.text}")
        return None

# TODO: need to figure out how to share the new created api key with the other division
# TODO: there really should be a send connection request endpoint (maybe that's when we create/share the key?)
def add_connection(source_division_id, target_division_id, daily_messages_count=0):
    # Generate API key and store hash
    raw_api_key = secrets.token_hex(32)
    hashed_api_key = hashlib.sha256(raw_api_key.encode()).hexdigest()
    
    data = {
        "source_division_id": source_division_id,
        "target_division_id": target_division_id,
        "daily_messages_count": daily_messages_count,
        "api_key": hashed_api_key
    }
    response = requests.post(f"{CENTRAL_API_BASE_URL}/connections", json=data, headers=headers)
    if is_2xx_status_code(response.status_code):
        result = response.json()
        # Securely send raw_api_key to the source division
        # For demonstration, return it
        return result['id'], raw_api_key
    else:
        logger.error(f"Error adding connection: {response.status_code} - {response.text}")
        return None, None

def find_connection(sender_division_id, receiver_division_id):
    params = {
        'division_id_1': sender_division_id,
        'division_id_2': receiver_division_id
    }
    response = requests.get(f"{CENTRAL_API_BASE_URL}/connections", params=params, headers=headers)
    if is_2xx_status_code(response.status_code):
        connections = response.json()
        if connections:
            return connections[0]
        else:
            return None  # No connection found
    else:
        error_message = response.json().get('error', 'Unknown error')
        logger.error(f"Error querying central database: {response.status_code} - {error_message}")
        return None  # Early exit on error

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
    if result.data:
        return result.data
    else:
        logger.error("Error: unable to insert custom data policy into data table.")
        return None

def get_division_api_url(division_id):
    # Query the central database via API to get the division's API URL
    response = requests.get(f"{CENTRAL_API_BASE_URL}/divisions/{division_id}/api_url", headers=headers)
    if is_2xx_status_code(response.status_code):
        data = response.json()
        return data.get('api_url')
    else:
        error_message = response.json().get('error', 'Unknown error')
        logger.error(f"Error getting division API URL: {response.status_code} - {error_message}")
        return None
