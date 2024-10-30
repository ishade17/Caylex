#!/bin/bash

curl -X POST http://localhost:8000/threads \
  -H "Content-Type: application/json" \
  -d '{
        "connection_id": 1,
        "messages_count": 0,
        "last_message_timestamp": "2024-01-01T12:00:00",
        "source_division_cost": 0.0,
        "target_division_cost": 0.0
      }'

# {
#   "thread_id": 1,
#   "connection_id": 1,
#   "messages_count": 0,
#   "last_message_timestamp": "2024-01-01T12:00:00",
#   "source_division_cost": 0.0,
#   "target_division_cost": 0.0
# }

curl -X GET http://localhost:8000/threads/1

# {
#   "thread_id": 1,
#   "connection_id": 1,
#   "messages_count": 0,
#   "last_message_timestamp": "2024-01-01T12:00:00",
#   "source_division_cost": 0.0,
#   "target_division_cost": 0.0
# }

curl -X PUT http://localhost:8000/threads/1 \
  -H "Content-Type: application/json" \
  -d '{
        "messages_count": 10,
        "source_division_cost": 5.0,
        "target_division_cost": 5.0
      }'

# {
#   "thread_id": 1,
#   "connection_id": 1,
#   "messages_count": 10,
#   "last_message_timestamp": "2024-01-01T12:00:00",
#   "source_division_cost": 5.0,
#   "target_division_cost": 5.0
# }

curl -X GET "http://localhost:8000/llm_context_windows?model_provider=openai&model_name=GPT-4"

# {
#   "id": 1,
#   "model_provider": "OpenAI",
#   "model_name": "gpt-4",
#   "context_window_length": 8192
# }

curl -X GET http://localhost:8000/divisions/tag/consulting_firm_acct

# {
#   "id": 1,
#   "company_id": 1,
#   "name": "Sales Division",
#   "tag": "sales_division",
#   "api_url": "http://localhost:8001"
# }

curl -X GET http://localhost:8000/divisions/1

# {
#   "id": 1,
#   "company_id": 1,
#   "name": "Customer Success",
#   "tag": "saas_customer_success",
#   "api_url": "http://localhost:8001"
# }

curl -X POST http://localhost:8000/divisions \
  -H "Content-Type: application/json" \
  -d '{
        "company_id": 1,
        "name": "Marketing Division",
        "tag": "marketing_division",
        "api_url": "http://localhost:8002"
      }'

# {
#   "id": 4,
#   "company_id": 1,
#   "name": "Marketing Division",
#   "tag": "marketing_division",
#   "api_url": "http://localhost:8002"
# }

curl -X GET http://localhost:8000/divisions/1/api_url

# {
#   "api_url": "http://localhost:8001"
# }

curl -X POST http://localhost:8000/companies \
  -H "Content-Type: application/json" \
  -d '{
        "name": "Tech Corp"
      }'

# {
#   "id": 4,
#   "name": "Tech Corp"
# }

curl -X GET http://localhost:8000/companies/4

# {
#   "id": 4,
#   "name": "Tech Corp"
# }

curl -X POST http://localhost:8000/connections \
  -H "Content-Type: application/json" \
  -d '{
        "source_division_id": 1,
        "target_division_id": 3
      }'

# {
#   "id": 1,
#   "source_division_id": 1,
#   "target_division_id": 3,
#   "daily_messages_count": 0
# }

curl -X GET "http://localhost:8000/connections?division_id_1=1&division_id_2=3"

# [
#   {
#     "id": 1,
#     "source_division_id": 1,
#     "target_division_id": 3,
#     "daily_messages_count": 0
#   }
# ]

### ERROR TESTS ###

curl -X POST http://localhost:8000/companies \
  -H "Content-Type: application/json" \
  -d '{}'

# {
#   "error": "'name' is required"
# }

curl -X POST http://localhost:8000/divisions \
  -H "Content-Type: application/json" \
  -d '{
        "company_id": "one",
        "name": "Finance Division",
        "tag": "finance_division",
        "api_url": "http://localhost:8003"
      }'

# {
#   "error": "'company_id' must be an integer"
# }

