import hashlib

from flask import Flask, request, jsonify 
from concurrent.futures import ThreadPoolExecutor

from agent_comm_dist import process_message
from global_vars import supabase
from send_message import auto_send_message
from present_message_with_context import present_message_with_context


### RECEIVE MESSAGES ###

app = Flask(__name__)

executor = ThreadPoolExecutor(max_workers=10)

# Endpoint to receive and store messages
@app.route('/message/receive', methods=['POST'])
def receive_message():
    try:
        # Extract API Key from headers
        api_key = request.headers.get('X-API-KEY')
        if not api_key:
            return jsonify({"error": "Missing API Key"}), 401
        
        hashed_api_key = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Extract connection_id from the message entry
        data = request.get_json()
        message_entry = data.get('message_entry')
        if not message_entry:
            return jsonify({"error": "Invalid data"}), 400
        connection_id = message_entry.get('connection_id')
        if not connection_id:
            return jsonify({"error": "Missing 'connection_id' in message entry"}), 400
        
        # Validate API Key against the specific connection
        result = supabase.table("connections") \
            .select("id") \
            .eq("id", connection_id) \
            .eq("api_key", hashed_api_key) \
            .execute()
        
        if not result.data:
            return jsonify({"error": "Invalid API Key for the specified connection"}), 403
        
        required_fields = ['connection_id', 'sender_division_id', 'receiver_division_id', 'message_content', 
                           'timestamp', 'status', 'thread_id', 'thread_msg_ordering', 'token_counts']
        for field in required_fields:
            if field not in message_entry:
                return jsonify({"error": f"'{field}' is required"}), 400

        # Insert message into Supabase
        message_entry['status'] = 'received'
        result = supabase.table("messages").insert(message_entry).execute()

        # Process message asynchronously with context
        executor.submit(process_message, message_entry)

        return jsonify({"message_id": result.data[0]['message_id']}), 200

    except Exception as e:
        app.logger.error(f"Error receiving message: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


### SEND AND PRESENT MESSAGES ###

@app.route('/message/auto_send', methods=['POST'])
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

@app.route('/message/present_with_context', methods=['POST'])
def api_present_message_with_context():
    data = request.get_json()
    message_info = data.get('message_info')
    model_provider = data.get('model_provider')
    model_name = data.get('model_name')
    if not message_info or not model_provider or not model_name:
        return jsonify({"error": "Invalid data"}), 400
    context = present_message_with_context(message_info, model_provider, model_name)
    return jsonify({'context': context})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)