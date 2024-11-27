import hashlib
import logging

from flask import Flask, request, jsonify 
from concurrent.futures import ThreadPoolExecutor

from agent_comm_dist import process_message, start_new_thread
from global_vars import supabase, logger
from connection_request_management import add_connection_to_local_database, create_api_key
from send_message import auto_send_message
from present_message_with_context import present_message_with_context


### RECEIVE MESSAGES ###

app = Flask(__name__)

app.logger.handlers = logger.handlers
app.logger.setLevel(logger.level)

executor = ThreadPoolExecutor(max_workers=10)

# Endpoint to receive and store messages
@app.route('/message/receive', methods=['POST'])
def receive_message():
    logger.info("/message/receive endpoint hit")
    print("/message/receive endpoint hit")
    try:
        # Extract API Key from headers
        raw_api_key = request.headers.get('X-API-KEY')
        if not raw_api_key:
            return jsonify({"error": "Missing API Key"}), 401
        
        # we are storing the raw api key in the local database and hashed one in the central database
        # hashed_api_key = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Extract connection_id from the message entry
        data = request.get_json()
        message_entry = data.get('message_entry')
        if not message_entry:
            return jsonify({"error": "Invalid data"}), 400
        
        required_fields = ['connection_id', 
                           'sender_division_id', 
                           'sender_division_tag', 
                           'sender_company_name',
                           'receiver_division_id', 
                           'receiver_division_tag', 
                           'receiver_company_name',
                           'message_content', 
                           'timestamp', 
                           'status', 
                           'thread_id', 
                           'thread_msg_ordering', 
                           'token_counts']
        for field in required_fields:
            if field not in message_entry:
                return jsonify({"error": f"'{field}' is required"}), 400
            
        # We use the central connection id for connection_id because it is standardized across the connected divisions.
        # This is different from other local data tables where we use the local connection id.
        connection_id = message_entry.get('connection_id')
        
        # Validate API Key against the specific connection
        result = supabase.table("connections") \
            .select("*") \
            .eq("central_connection_id", connection_id) \
            .eq("raw_api_key", raw_api_key) \
            .execute()
        
        if not result.data:
            return jsonify({"error": "Invalid API Key for the specified central connection id"}), 403

        # Insert message into Supabase
        message_entry['status'] = 'received'
        result = supabase.table("messages").insert(message_entry).execute()

        logger.info("Within receive_message(), about to execute process_message function")
        print("Within receive_message(), about to execute process_message function")

        # Process message asynchronously with context
        # process_message() calls auto_send_message(), which calls the receive_message() endpoint.
        # this is okay because this is the messsaging loop, but we cant have this for testing.
        # executor.submit(process_message, message_entry)
        process_message(message_entry)

        logger.info("Within receive_message(), finished with the process_message function")
        print("Within receive_message(), finished with the process_message function")
        return jsonify({"message_id": result.data[0]['message_id']}), 200

    except Exception as e:
        app.logger.error(f"Error receiving message: {e}")
        return jsonify({"error": "Internal Server Error"}), 500
    

### START A NEW THREAD ###

@app.route('/message/new_thread', methods=['POST'])
def new_thread():
    data = request.get_json()
    message = data.get('message')
    logger.info(f"Within new_thread(): {data}")
    print(f"Within new_thread(): {data}")
    if not message:
        return jsonify({"error": "No message data"}), 400
    
    logger.info("start_new_thread() called synchronously")
    print("start_new_thread() called synchronously")
    start_new_thread(message)

    try:
        logger.info("start_new_thread() called asynchronously")
        print("start_new_thread() called asynchronously")
        future = executor.submit(start_new_thread, message)
        logger.info("Task submitted to ThreadPoolExecutor")
        print("Task submitted to ThreadPoolExecutor")
    except Exception as e:
        logger.error(f"Error submitting task to ThreadPoolExecutor: {e}")
        print(f"Error submitting task to ThreadPoolExecutor: {e}")

    # executor.submit(start_new_thread, message)
    return jsonify({"message": "New thread started"}), 200

### RECEIVE CONNECTION REQUESTS ###

@app.route('/connection/request/accept', methods=['POST'])
def accept_connection_request():
    data = request.get_json()
    sender_division_id = data.get('sender_division_id')
    # TODO: figure out how to wait for user to accept connection (this will eventually come from a UI interaction)
    # This logic^ will involve the sender_division_id etc.
    connection_request_response = True
    if connection_request_response:
        raw_api_key, hashed_api_key = create_api_key()
        return jsonify({'response': 'success', 'raw_api_key': raw_api_key, 'hashed_api_key': hashed_api_key}), 200
    else:
        return jsonify({'response': 'error', 'message': 'Missing sender_division_id'}), 400
    
@app.route('/connection/request/insert', methods=['POST'])
def insert_connection_request():
    data = request.get_json()
    sender_division_id = data.get('sender_division_id')
    current_division_id = data.get('current_division_id')
    raw_api_key = data.get('raw_api_key')
    central_connection_id = data.get('central_connection_id')

    result = add_connection_to_local_database(sender_division_id, current_division_id, raw_api_key, central_connection_id)
    if result:
        return jsonify({'local_connection_id': result}), 200
    else:
        return jsonify({'response': 'error', 'message': 'Failed to insert connection request'}), 500


### SEND AND PRESENT MESSAGES ###

@app.route('/message/auto_send', methods=['POST'])
def api_auto_send_message():
    logger.info("/message/auto_send endpoint hit")
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

# Be sure to change port when copied over to Enterprise Division 2
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002)


# TODO: first test out the new_thread endpoint -- make sure auto_send is executing (it won't actually send until next step...)
# then try running communication across two divisions -- you'll need to start up: 
# - division-api-server from enterprise division 1 (port 8002) + kong (port 8000) + customer agent 1 (port 5001), 
# - division-api-server from enterprise division 2 (port 5003) + kong (port 8001) + customer agent 2 (port 5004)
# - central-api-server (port 8001)
# we also need to add an end condition on sending messages

"""
test the agent communication api endpoints from terminal:

docker compose exec db psql -U postgres -d postgres

INSERT INTO connections (
    central_connection_id,
    source_division_id,
    target_division_id,
    daily_messages_count,
    raw_api_key,
    created_at
) VALUES (
    1,  -- central_connection_id
    1,  -- source_division_id
    2,  -- target_division_id
    0,  -- daily_messages_count
    '8d625c78b565f0b3d6684c10a4660a2f8c6e88bcffc2aebebade1862dab945ff',  -- raw API key
    now()
);


SELECT * FROM connections;

# new message in new thread
# this is a message incoming to saas_customer_success (the division we're testing on)
docker exec division-api-server curl -X POST http://division-api:8002/message/receive \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: 8d625c78b565f0b3d6684c10a4660a2f8c6e88bcffc2aebebade1862dab945ff" \
  -d '{
    "message_entry": {
      "connection_id": 1,
      "sender_division_id": 2,
      "sender_division_tag": "consulting_firm_acct",
      "sender_company_name": "Consulting Firm",
      "receiver_division_id": 1,
      "receiver_division_tag": "saas_customer_success",
      "receiver_company_name": "SaaS Solutions",
      "message_content": "@saas_customer_success Hello, World! @new_thread",
      "timestamp": "2024-11-08T12:00:00Z",
      "status": "sent",
      "thread_id": 2,
      "thread_msg_ordering": 1,
      "token_counts": {}
    }
  }'


# new message in existing thread
# this is a message incoming to saas_customer_success (the division we're testing on)
docker exec division-api-server curl -X POST http://division-api:8002/message/receive \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: 8d625c78b565f0b3d6684c10a4660a2f8c6e88bcffc2aebebade1862dab945ff" \
  -d '{
    "message_entry": {
      "connection_id": 1,
      "sender_division_id": 2,
      "sender_division_tag": "consulting_firm_acct",
      "sender_company_name": "Consulting Firm",
      "receiver_division_id": 1,
      "receiver_division_tag": "saas_customer_success",
      "receiver_company_name": "SaaS Solutions",
      "message_content": "@saas_customer_success second message @thread_6",
      "timestamp": "2024-11-08T12:01:00Z",
      "status": "sent",
      "thread_id": 6,
      "thread_msg_ordering": 2,
      "token_counts": {}
    }
  }'

  

# this is a message being sent out from saas_customer_success (the division we're testing on) on a new thread
docker exec division-api-server curl -X POST http://division-api:8002/message/new_thread \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: 8d625c78b565f0b3d6684c10a4660a2f8c6e88bcffc2aebebade1862dab945ff" \
  -d '{
    "message": "@consulting_firm_acct please devise a plan for building a custom data pipeline that intakes news articles and outputs summaries @new_thread"
}'

# this is a message being sent out from saas_customer_success (the division we're testing on) on an existing thread
docker exec division-api-server curl -X POST http://division-api:8002/message/new_thread \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: 8d625c78b565f0b3d6684c10a4660a2f8c6e88bcffc2aebebade1862dab945ff" \
  -d '{
    "message": "@consulting_firm_acct second message @thread_6"
}'



"""