import secrets
import hashlib
import os
import logging

from flask import Flask, request, jsonify, abort 
from supabase import create_client, Client
from datetime import datetime
from functools import wraps

app = Flask(__name__)

# Initialize Supabase client for the central database
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get API key from Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Invalid or missing Authorization header"}), 401
        api_key = auth_header[len('Bearer '):]
        hashed_api_key = hashlib.sha256(api_key.encode()).hexdigest()
        # Validate the API key against the divisions table
        result = supabase.table("divisions").select("id").eq("api_key", hashed_api_key).execute()
        if not result.data:
            return jsonify({"error": "Invalid API Key"}), 403
        # Optionally, set division info in the request context
        request.division_id = result.data[0]['id']
        return f(*args, **kwargs)
    return decorated_function

@app.route('/threads', methods=['POST'])
@require_api_key
def create_thread():
    """
    Create a new collaboration thread.
    Expects JSON payload with required fields.
    Returns the created thread object.
    """
    data = request.get_json()

    # Validate required fields
    required_fields = ['connection_id', 'messages_count', 'last_message_timestamp', 'source_division_cost', 'target_division_cost']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"'{field}' is required"}), 400

    # TODO: we are failing somewhere in here...
    app.logger.info(f"thread data: {data}")
    logger.info(f"thread data: {data}")
    print(f"thread data: {data}")

    # Validate data types
    if not isinstance(data['connection_id'], int):
        return jsonify({"error": "'connection_id' must be an integer"}), 400
    if not isinstance(data.get('messages_count', 0), int):
        return jsonify({"error": "'messages_count' must be an integer"}), 400
    try:
        # Validate 'last_message_timestamp' format
        # Check if microseconds are present
        if "." in data['last_message_timestamp']:
            # Parse with microseconds and remove them
            timestamp = datetime.strptime(data['last_message_timestamp'], '%Y-%m-%dT%H:%M:%S.%f')
        else:
            # Parse without microseconds
            timestamp = datetime.strptime(data['last_message_timestamp'], '%Y-%m-%dT%H:%M:%S')
        # Standardize the timestamp format (without microseconds)
        data['last_message_timestamp'] = timestamp.strftime('%Y-%m-%dT%H:%M:%S')
    except ValueError:
        app.logger.error(f"Error: 'last_message_timestamp' is not in ISO format: {data['last_message_timestamp']}")
        logger.error(f"Error: 'last_message_timestamp' is not in ISO format: {data['last_message_timestamp']}")
        return jsonify({"error": "'last_message_timestamp' must be in ISO format YYYY-MM-DDTHH:MM:SS"}), 400
    except KeyError:
        app.logger.error(f"Error: 'last_message_timestamp' is required")
        logger.error(f"Error: 'last_message_timestamp' is required")
        return jsonify({"error": "'last_message_timestamp' is required"}), 400

    try:
        data['source_division_cost'] = float(data['source_division_cost'])
        data['target_division_cost'] = float(data['target_division_cost'])
    except ValueError:
        app.logger.error(f"Error: 'source_division_cost' and 'target_division_cost' must be numbers: {data['source_division_cost']} and {data['target_division_cost']}")
        logger.error(f"Error: 'source_division_cost' and 'target_division_cost' must be numbers: {data['source_division_cost']} and {data['target_division_cost']}")
        return jsonify({"error": "'source_division_cost' and 'target_division_cost' must be numbers"}), 400

    try:
        result = supabase.table("collab_threads").insert(data).execute()
        if not result.data:
            app.logger.error(f"Error: Unable to create thread")
            logger.error(f"Error: Unable to create thread")
            return jsonify({"error": "Unable to create thread"}), 400
        return jsonify(result.data[0]), 201  # Return 201 Created
    except Exception as e:
        app.logger.error(f"Error creating thread: {e}")
        logger.error(f"Error creating thread: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/threads/<int:thread_id>', methods=['GET'])
@require_api_key
def get_thread(thread_id):
    """
    Retrieve a collaboration thread by its ID.
    Returns the thread object.
    """
    try:
        result = supabase.table("collab_threads").select("*").eq("thread_id", thread_id).execute()
        if not result.data:
            logger.error(f"Error: Thread not found")
            abort(404, description="Thread not found")
        return jsonify(result.data[0]), 200
    except Exception as e:
        app.logger.error(f"Error fetching thread: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/threads/<int:thread_id>', methods=['PUT'])
@require_api_key
def update_thread(thread_id):
    """
    Update a collaboration thread by its ID.
    Expects JSON payload with fields to update.
    Returns the updated thread object.
    """
    data = request.get_json()
    try:
        result = supabase.table("collab_threads").update(data).eq("thread_id", thread_id).execute()
        if not result.data:
            abort(404, description="Thread not found")
        return jsonify(result.data[0]), 200
    except Exception as e:
        app.logger.error(f"Error updating thread: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/llm_context_windows', methods=['GET'])
@require_api_key
def get_llm_context_windows():
    """
    Retrieve LLM context window information based on model provider and name.
    Expects 'model_provider' and 'model_name' as query parameters.
    Returns the matching context window.
    """
    model_provider = request.args.get('model_provider')
    model_name = request.args.get('model_name')
    if not model_provider or not model_name:
        return jsonify({"error": "'model_provider' and 'model_name' are required parameters."}), 400
    try:
        result = supabase.table("llm_context_windows").select("*").eq("model_provider", model_provider).eq("model_name", model_name).execute()
        if not result.data:
            abort(404, description="Context window not found")
        logger.info(f"Successfully fetched LLM context windows for {model_provider} {model_name}")
        return jsonify(result.data[0]), 200
    except Exception as e:
        app.logger.error(f"Error fetching LLM context windows: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# GET /divisions/tag/<receiver_division_tag>
@app.route('/divisions/tag/<string:division_tag>', methods=['GET'])
@require_api_key
def get_division_by_tag(division_tag):
    """
    Retrieve a division by its tag.
    Returns the division object.
    """
    try:
        result = supabase.table("divisions").select("*").eq("tag", division_tag).execute()
        if not result.data:
            abort(404, description="Division not found")
        return jsonify(result.data[0]), 200
    except Exception as e:
        app.logger.error(f"Error fetching division by tag: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# GET /divisions/<int:division_id>
@app.route('/divisions/<int:division_id>', methods=['GET'])
@require_api_key
def get_division_by_id(division_id):
    """
    Retrieve a division by its ID.
    Returns the division object.
    """
    try:
        result = supabase.table("divisions").select("*").eq("id", division_id).execute()
        if not result.data:
            abort(404, description="Division not found")
        return jsonify(result.data[0]), 200
    except Exception as e:
        app.logger.error(f"Error fetching division by ID: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# POST /divisions/insert
@app.route('/divisions/insert', methods=['POST'])
@require_api_key
def insert_division():
    """
    Insert a new division.
    Expects JSON payload with required fields.
    Returns the created division object.
    """
    data = request.get_json()
    required_fields = ['company_id', 'name', 'tag', 'api_url']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"'{field}' is required"}), 400
        
    # Validate company_id as integer
    if not isinstance(data['company_id'], int):
        return jsonify({"error": "'company_id' must be an integer"}), 400
    
     # Generate API key
    raw_api_key = secrets.token_hex(32)  # 64-character hex string
    hashed_api_key = hashlib.sha256(raw_api_key.encode()).hexdigest()
    data['api_key'] = hashed_api_key  # Store the hashed API key

    try:
        result = supabase.table("divisions").insert(data).execute()
        if not result.data:
            return jsonify({"error": "Unable to insert division"}), 400
        # Securely send the raw API key back to the division
        division_data = result.data[0]
        division_data['api_key'] = raw_api_key  # Include raw API key in the response
        return jsonify(result.data[0]), 201
    except Exception as e:
        app.logger.error(f"Error inserting division: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# GET /divisions/<int:division_id>/api_url
@app.route('/divisions/<int:division_id>/api_url', methods=['GET'])
@require_api_key
def get_division_api_url(division_id):
    """
    Retrieve the API URL of a division by its ID.
    Returns the API URL.
    """
    try:
        result = supabase.table("divisions").select("api_url").eq("id", division_id).execute()
        if not result.data:
            abort(404, description="Division not found")
        return jsonify({"api_url": result.data[0]['api_url']}), 200
    except Exception as e:
        app.logger.error(f"Error fetching division API URL: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# POST /companies
@app.route('/companies', methods=['POST'])
@require_api_key
def insert_company():
    """
    Insert a new company.
    Expects JSON payload with required fields.
    Returns the created company object.
    """
    data = request.get_json()
    required_fields = ['name']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"'{field}' is required"}), 400
    try:
        result = supabase.table("companies").insert(data).execute()
        if not result.data:
            return jsonify({"error": "Unable to insert company"}), 400
        return jsonify(result.data[0]), 201
    except Exception as e:
        app.logger.error(f"Error inserting company: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# GET /companies/<int:company_id>
@app.route('/companies/<int:company_id>', methods=['GET'])
@require_api_key
def get_company_by_id(company_id):
    """
    Retrieve a company by its ID.
    Returns the company object.
    """
    try:
        result = supabase.table("companies").select("*").eq("id", company_id).execute()
        if not result.data:
            abort(404, description="Company not found")
        return jsonify(result.data[0]), 200
    except Exception as e:
        app.logger.error(f"Error fetching company by ID: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# POST /connections/insert
@app.route('/connections/insert', methods=['POST'])
@require_api_key
def insert_connection():
    """
    Insert a new connection between divisions.
    Expects JSON payload with required fields.
    Returns the created connection object.
    """
    data = request.get_json()
    required_fields = ['source_division_id', 'target_division_id']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"'{field}' is required"}), 400
    # Validate that division IDs are integers
    try:
        data['source_division_id'] = int(data['source_division_id'])
        data['target_division_id'] = int(data['target_division_id'])
    except ValueError:
        return jsonify({"error": "'source_division_id' and 'target_division_id' must be integers"}), 400

    if "daily_messages_count" not in data:
        data["daily_messages_count"] = 0  # Set default value

    try:
        result = supabase.table("connections").insert(data).execute()
        if not result.data:
            return jsonify({"error": "Unable to insert connection"}), 400
        return jsonify(result.data[0]), 201
    except Exception as e:
        app.logger.error(f"Error inserting connection: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# GET /connections
@app.route('/connections', methods=['GET'])
@require_api_key
def get_connections():
    """
    Retrieve connections between two divisions.
    Expects 'division_id_1' and 'division_id_2' as query parameters.
    Returns a list of connections.
    """
    division_id_1 = request.args.get('division_id_1')
    division_id_2 = request.args.get('division_id_2')

    if not division_id_1 or not division_id_2:
        return jsonify({"error": "division_id_1 and division_id_2 are required parameters."}), 400

    # Validate that division IDs are integers
    try:
        division_id_1 = int(division_id_1)
        division_id_2 = int(division_id_2)
    except ValueError:
        return jsonify({"error": "division_id_1 and division_id_2 must be integers."}), 400

    try:
        # Construct the filter string using correct syntax
        filter_string = (
            f"and(source_division_id.eq.{division_id_1},target_division_id.eq.{division_id_2}),"
            f"and(source_division_id.eq.{division_id_2},target_division_id.eq.{division_id_1})"
        )

        # Query connections where the two divisions are connected, regardless of order
        result = supabase.table("connections").select("*").or_(filter_string).execute()
        
        if not result.data:
            return jsonify([]), 200  # Return an empty list if no connections found

        return jsonify(result.data), 200
    except Exception as e:
        print("error")
        app.logger.error(f"Error fetching connections: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run()
