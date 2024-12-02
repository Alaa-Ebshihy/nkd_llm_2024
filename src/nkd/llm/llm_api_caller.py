"""
functions to call API for different LLMs
"""
from openai import OpenAI
from together import Together

LLM_NAME_MAP = {
    'gpt4o_mini': 'gpt-4o-mini-2024-07-18',
    'gpt4o': 'gpt-4o-2024-05-13',
    'llama31-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
    'llama31-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
}


def get_llm_response(api_type, prompt, llm_model, api_key):
    """
    :param api_type: can be together or open_ai
    :param prompt:
    :param llm_model: the name of the model recognized by the api
    :param api_key:
    :return:
    """
    if api_type == 'together':
        return get_together_response(prompt, LLM_NAME_MAP[llm_model], api_key)
    return get_open_ai_response(prompt, LLM_NAME_MAP[llm_model], api_key)


def get_together_response(prompt, model_name, api_key):
    client = Together(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            },
        ],
        temperature=0.7,
        top_p=1
    ).choices[0].message.content

    return response


def get_open_ai_response(prompt, model_name, api_key):
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            },
        ],
        temperature=0.7,
        max_tokens=1024,
        top_p=1
    ).choices[0].message.content

    return response
