import tiktoken

from vertexai.preview import tokenization

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from transformers import LlamaTokenizer
from transformers import AutoTokenizer

### COUNT MESSAGE TOKENS ###

# Define tokenizer functions
def count_tokens_openai(text, model):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def count_tokens_meta(text, model):
    tokenizer = LlamaTokenizer.from_pretrained(model)
    return len(tokenizer.encode(text))

def count_tokens_anthropic(text, model):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def count_tokens_google(text, model):
    tokenizer = tokenization.get_tokenizer_for_model(model)
    result = tokenizer.count_tokens(text)
    return result.total_tokens

def count_tokens_mistral(text, model):
    tokenizer = MistralTokenizer.from_model(model)
    chat_request = ChatCompletionRequest(messages=[UserMessage(content=text)], model=model)
    tokenized = tokenizer.encode_chat_completion(chat_request)
    return len(tokenized.tokens)

def count_tokens_cohere(text, model):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def count_tokens_with_auto_tokenizer(text, model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    return len(tokenizer.encode(text))

# Default models for various LLM providers
# TODO: maybe also throw this in a data table (i'll do this later -- lower priority)
default_models = {
    'openai': 'gpt-4-turbo',
    # 'meta': 'meta-llama/Llama-2-7b-hf',
    # 'anthropic': 'claude-3',
    'anthropic': 'gpt-4-turbo',
    'google': 'gemini-1.5-pro-001',
    'mistral': 'open-mixtral-8x22b',
    # 'cohere': 'cohere-command'
    'cohere': 'gpt-4-turbo'
}

# Tokenizers dictionary with a simplified lambda that supports passing a model name
tokenizers = {
    'openai': count_tokens_openai,
    'meta': count_tokens_meta,
    'anthropic': count_tokens_anthropic,
    'google': count_tokens_google,
    'mistral': count_tokens_mistral,
    'cohere': count_tokens_cohere,
}