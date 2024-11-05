import argparse
import requests
from flask import request, jsonify 

from global_vars import supabase, DIVISION_TAG, is_2xx_status_code, CENTRAL_API_BASE_URL, CENTRAL_API_KEY
from database_interaction import add_connection, create_api_key, get_division_id_from_tag

### MANAGE CONNECTION REQUESTS ###

def add_connection_to_local_database(sender_division_id, current_division_id, raw_api_key, central_connection_id):
    connection_entry = {
        'source_division_id': sender_division_id,
        'target_division_id': current_division_id,
        'daily_messages_count': 0,
        'central_connection_id': central_connection_id,
        'api_key': raw_api_key
    }
    result = supabase.table("connections").insert(connection_entry).execute()
    if result.data:
        return result.data[0]['id']
    else:
        raise Exception("Error inserting connection to local database")

def send_connection_request(target_division_tag):
    # Send request to target division
    current_division_id = get_division_id_from_tag(DIVISION_TAG)
    target_division_id = get_division_id_from_tag(target_division_tag)
    # Send request to target division
    headers = {'Authorization': f'Bearer {CENTRAL_API_KEY}'}
    target_division_api_url = requests.get(f"{CENTRAL_API_BASE_URL}/divisions/{target_division_id}/api_url", headers=headers)
    connection_request_data = {'sender_division_id': current_division_id}
    response = requests.post(f"{target_division_api_url}/connection/request/accept", json=connection_request_data)
    # Add connection to central database and local databases
    if is_2xx_status_code(response.status_code):
        data = response.json()
        raw_api_key = data.get('raw_api_key')
        hashed_api_key = data.get('hashed_api_key')
        # Add connection to central database
        central_connection_id = add_connection(current_division_id, target_division_id, hashed_api_key, 0)
        # Insert connection in target division database
        connection_insert_data = {'source_division_id': current_division_id, 
                                  'target_division_id': target_division_id, 
                                  'raw_api_key': raw_api_key,
                                  'central_connection_id': central_connection_id}
        insert_connection_response = requests.post(f"{target_division_api_url}/connection/request/insert", json=connection_insert_data)
        if is_2xx_status_code(insert_connection_response.status_code):
            # Add connection to local database
            local_connection_id = add_connection_to_local_database(current_division_id, target_division_id, raw_api_key, central_connection_id)
            return jsonify({'response': 'success', 'local_connection_id': local_connection_id}), 200
        else:
            return jsonify({'response': 'error', 'message': 'Failed to insert connection request'}), 500
    else:
        return jsonify({'response': 'error', 'message': 'Failed to accept connection request'}), 500

### CLI Main Function ###

def main():
    parser = argparse.ArgumentParser(description="CLI for managing connection requests.")
    subparsers = parser.add_subparsers(dest="command")

    # Command to send a connection request
    parser_send_request = subparsers.add_parser("send_request", help="Send a connection request to another division")
    parser_send_request.add_argument("target_division_tag", help="Tag of the target division for the connection request")

    args = parser.parse_args()

    # Execute based on CLI command
    if args.command == "send_request":
        result = send_connection_request(args.target_division_tag)
        if result.get('response') == 'success':
            print(f"Connection request successful. Local connection ID: {result['local_connection_id']}")
        else:
            print(f"Error: {result.get('message', 'Unknown error occurred')}")

if __name__ == "__main__":
    main()