import os
import re

from supabase import create_client, Client
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import logging
from dotenv import load_dotenv 


### GLOBAL VARIABLES ###

# parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load the .env file from the parent directory
# dotenv_path = os.path.join(parent_directory, ".env")
# load_dotenv(dotenv_path)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Print all environment variables
for key, value in os.environ.items():
    print(f"{key}: {value}")
    logger.debug(f"{key}: {value}")


# Connect to Supabase
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')

if not supabase_url:
    raise EnvironmentError("SUPABASE_URL must be set in environment variables.")
if not supabase_key:
    raise EnvironmentError("SUPABASE_KEY/SERVICE_ROLE_KEY must be set in environment variables.")


supabase: Client = create_client(supabase_url, supabase_key)

AGENT_COMM_API_BASE_URL = os.getenv('AGENT_COMM_API_BASE_URL')
CUSTOMER_AGENT_BASE_URL = os.getenv('CUSTOMER_AGENT_BASE_URL')

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_KEY"),
)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

genai.configure(api_key=os.getenv("GOOGLE_KEY"))

company_name = os.getenv('COMPANY_NAME')

### API URLS ###

CENTRAL_API_BASE_URL = os.getenv('CENTRAL_API_BASE_URL')
CENTRAL_API_KEY = os.getenv('CENTRAL_API_KEY')
if not CENTRAL_API_BASE_URL or not CENTRAL_API_KEY:
    raise EnvironmentError("CENTRAL_API_URL and CENTRAL_API_KEY must be set in environment variables.")


### API RESPONSE CHECKER ###

def is_2xx_status_code(status_code):
    return re.match(r"^2\d{2}$", str(status_code))
