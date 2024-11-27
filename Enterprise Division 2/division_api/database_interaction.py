import requests

# TODO: need to add to requirements.txt ?
import argparse


from global_vars import supabase, logger, CENTRAL_API_BASE_URL, CENTRAL_API_KEY, is_2xx_status_code

### CENTRAL DATABASE INTERACTION ###

# All these functions will likely be triggered via UI actions so no need to expose them as api endpoints

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {CENTRAL_API_KEY}",
}   

# TODO: need to add a function that copies all the relevant data for a specific company from the central database into the local database
# this is necessary for when we test endpoints because the fake companies need to know their connections etc.
# we also need to actually generate the real api_keys and insert that into both the central and local databases

def register_company(name):
    data = {"name": name}
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

def add_connection(source_division_id, target_division_id, hashed_api_key, daily_messages_count=0):
    data = {
        "source_division_id": source_division_id,
        "target_division_id": target_division_id,
        "daily_messages_count": daily_messages_count,
        "api_key": hashed_api_key
    }
    response = requests.post(f"{CENTRAL_API_BASE_URL}/connections", json=data, headers=headers)
    if is_2xx_status_code(response.status_code):
        result = response.json()
        return result['id']
    else:
        logger.error(f"Error adding connection: {response.status_code} - {response.text}")
        return None, None
    
def get_local_connection_id(central_connection_id):
    result = supabase.table("connections").select("local_connection_id").eq("central_connection_id", central_connection_id).execute()
    if result.data:
        return result.data[0]['local_connection_id']
    else:
        return None

def find_connection(sender_division_id, receiver_division_id, hashed_api_key=""):
    approved_to_send_message = False
    params = {
        'division_id_1': sender_division_id,
        'division_id_2': receiver_division_id
    }
    response = requests.get(f"{CENTRAL_API_BASE_URL}/connections", params=params, headers=headers)
    if is_2xx_status_code(response.status_code):
        connections = response.json()
        if connections:
            connection = connections[0]
            if hashed_api_key == "":
                return connection, approved_to_send_message, None
            elif connection['api_key'] == hashed_api_key:
                approved_to_send_message = True
                local_connection_id = get_local_connection_id(connection['id'])
                return connection, approved_to_send_message, local_connection_id
            else:
                logger.error(f"API key does not match for connection between {sender_division_id} and {receiver_division_id}")
                return None, approved_to_send_message, None
        else:
            logger.error(f"No connection found between {sender_division_id} and {receiver_division_id}")
            return None, approved_to_send_message, None
    else:
        error_message = response.json().get('error', 'Unknown error')
        logger.error(f"Error querying central database: {response.status_code} - {error_message}")
        return None, approved_to_send_message, None

def insert_data_policy(sender_division_id, receiver_division_id, confidentiality, data_type, explanation):
    # Look up the connection using API call to central database
    connection, _, local_connection_id = find_connection(sender_division_id, receiver_division_id)
    if not connection:
        return "No valid connection found between the specified divisions."
    
    # connection_id = connection['id']

    # Insert data into the division's own 'custom_data_policies' table
    data = {
        "connection_id": local_connection_id,
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
    
def get_division_id_from_tag(division_tag):
    response = requests.get(f"{CENTRAL_API_BASE_URL}/divisions/tag/{division_tag}", headers=headers)
    if is_2xx_status_code(response.status_code):
        data = response.json()
        return data.get('id')
    else:
        logger.error(f"Error getting division ID: {response.status_code} - {response.text}")
        return None


### Main Function for CLI Interface ###

# For testing purposes
def main():
    parser = argparse.ArgumentParser(description="CLI for interacting with the central database.")
    subparsers = parser.add_subparsers(dest="command")

    # Register Company
    parser_register_company = subparsers.add_parser("register_company")
    parser_register_company.add_argument("name", help="Name of the company to register")

    # Register Division
    parser_register_division = subparsers.add_parser("register_division")
    parser_register_division.add_argument("company_id", help="Company ID to which the division belongs")
    parser_register_division.add_argument("name", help="Name of the division")
    parser_register_division.add_argument("tag", help="Tag for the division")

    # Add Connection
    parser_add_connection = subparsers.add_parser("add_connection")
    parser_add_connection.add_argument("source_division_id", help="Source division ID")
    parser_add_connection.add_argument("target_division_id", help="Target division ID")
    parser_add_connection.add_argument("--daily_messages_count", type=int, default=0, help="Daily message count (default 0)")

    # Find Connection
    parser_find_connection = subparsers.add_parser("find_connection")
    parser_find_connection.add_argument("sender_division_id", help="Sender division ID")
    parser_find_connection.add_argument("receiver_division_id", help="Receiver division ID")

    # Insert Data Policy
    parser_insert_data_policy = subparsers.add_parser("insert_data_policy")
    parser_insert_data_policy.add_argument("sender_division_id", help="Sender division ID")
    parser_insert_data_policy.add_argument("receiver_division_id", help="Receiver division ID")
    parser_insert_data_policy.add_argument("confidentiality", help="Confidentiality level")
    parser_insert_data_policy.add_argument("data_type", help="Type of data")
    parser_insert_data_policy.add_argument("explanation", help="Explanation of the data policy")

    # Get Division API URL
    parser_get_division_api_url = subparsers.add_parser("get_division_api_url")
    parser_get_division_api_url.add_argument("division_id", help="Division ID")

    args = parser.parse_args()

    # Call functions based on CLI command
    if args.command == "register_company":
        print(register_company(args.name))
    elif args.command == "register_division":
        print(register_division(args.company_id, args.name, args.tag))
    elif args.command == "add_connection":
        print(add_connection(args.source_division_id, args.target_division_id, args.daily_messages_count))
    elif args.command == "find_connection":
        print(find_connection(args.sender_division_id, args.receiver_division_id))
    elif args.command == "insert_data_policy":
        print(insert_data_policy(args.sender_division_id, args.receiver_division_id, args.confidentiality, args.data_type, args.explanation))
    elif args.command == "get_division_api_url":
        print(get_division_api_url(args.division_id))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()


"""

Examples:
python database_interaction.py register_company "New Company Name"
python database_interaction.py register_division "company_id" "Division Name" "Division Tag"
python database_interaction.py add_connection "source_division_id" "target_division_id" --daily_messages_count 10
python database_interaction.py find_connection "sender_division_id" "receiver_division_id"

"""