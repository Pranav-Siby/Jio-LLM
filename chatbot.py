import streamlit as st
import json
import asyncio
import requests
from vector_space_model import handle_query_by_vector_space_model
import ollama
from collections import deque

# Initialize session state for storing conversation history and last mentioned brand
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = deque(maxlen=5)  # Only store the last 5 prompts

if "last_mentioned_brand" not in st.session_state:
    st.session_state.last_mentioned_brand = ""


def classify_and_identify_brand(prompt):
    """Classify the prompt as relevant or irrelevant and identify brands if relevant."""
    classifier_prompt = f"""
    You are an assistant for a shopping application by the company Jio, called JioMart. Your task is to classify 
    the user's query as either 'relevant' if it relates to products available in JioMart or 'irrelevant' if it 
    contains gibberish or is not related to JioMart's products. If the query is relevant, identify any brands mentioned. 
    Here is the user query: '{prompt}'

    Return the classification and identified brand in the format 'classification: [relevant/irrelevant], brand: [brand name]'.
    If no brand is mentioned, return 'none' for the brand.
    """

    response = ollama.chat(model='llama3', messages=[
        {'role': 'system', 'content': classifier_prompt},
        {'role': 'user', 'content': prompt},
    ])
    return response['message']['content'].strip().lower()


def modify_prompt_with_last_brand(prompt, brand):
    """Modify the user's prompt to include the detected or last mentioned brand."""
    last_mentioned_brand = st.session_state.last_mentioned_brand

    if brand and brand != 'none':
        last_mentioned_brand = brand
        prompt = f"{brand} {prompt}"
    elif last_mentioned_brand:
        prompt = f"{last_mentioned_brand} {prompt}"

    st.session_state.last_mentioned_brand = last_mentioned_brand

    return prompt


def llm_api(prompt):
    """Call the LLM API with the given prompt and handle exceptions."""
    new_system_prompt = """
    You are a chatbot for a shopping application by the company Jio, called JioMart. Your task is to only respond to user
    queries regarding products in the shop. If the user gives you a greeting, greet them back.
    If the user asks for anything irrelevant or gives a gibberish prompt, kindly ask them to ask a question related to 
    the products in JioMart. Your task is only to give current information
    about the products. You cannot add items to cart or give the user notifications. Just answer the questions. 
    """

    full_prompt = f"{new_system_prompt}\n\nThis is the user's conversation history:\n\
                    {st.session_state.conversation_history}\n\nUser: {prompt}"

    full_prompt = f"{new_system_prompt}\n\nThis is the user's conversation history:{', '.join(st.session_state.conversation_history)}\n\nUser: {prompt}"

    try:
        response = ollama.chat(model='llama3', messages=[
            {'role': 'system', 'content': full_prompt},
            {'role': 'user', 'content': prompt},
        ])
        return response['message']['content']
    except requests.exceptions.RequestException as e:
        return f"Error communicating with the LLM API: {e}"
    except KeyError:
        return "Error: Invalid response from the LLM API."


def query_data(query):
    """Query the data using the vector space model and handle exceptions."""
    try:
        response = handle_query_by_vector_space_model(query)
        if not response:
            return []
        return response  # list of tuples of the form: (product_id, product_name, category, price, description, units, score)
    except Exception as e:
        print(f"Error querying data: {e}")
        return []


def get_category(query: str, category_list: list[str]):
    classifier_prompt = f'''You are a classifier for a shopping application by the company Jio, called JioMart. \
            Your task is to identify which of the following categories of products the user query is asking for. \ 
            The possible categories are {category_list}. \
            You must give a single word as your answer.'''
    response = ollama.chat(model='llama3', messages=[
        {'role': 'system', 'content': classifier_prompt},
        {'role': 'user', 'content': query},
    ])
    return response['message']['content']


def generate_response(msg: str):
    """Generate a response based on the user query and relevant data."""
    # Classify the prompt and identify any brands
    classification_response = classify_and_identify_brand(msg)
    classification, brand = classification_response.split(', ')
    classification = classification.split(': ')[1]
    brand = brand.split(': ')[1]

    if classification == 'irrelevant':
        return "Please ask a question related to the products in JioMart."

    # Modify the query with last mentioned brand if necessary
    modified_prompt = modify_prompt_with_last_brand(msg, brand)

    relevant_results = query_data(modified_prompt)
    if not relevant_results:
        final_prompt = modified_prompt + "\n(We do not have any such items in stock.)"
        return llm_api(final_prompt)

    category_list = ["shoes", "cosmetics", "electronics", "clothing", "accessories", "sports"]  # can be read from db
    correct_category = get_category(modified_prompt, category_list).lower()

    final_prompt = modified_prompt + "(Here is a list of relevant items we have and the info you should use to answer: "

    top = 3
    for result in relevant_results:
        if top > 0 and result[2] == correct_category:
            top -= 1
            product_id, product_name, category, price, description, units, score = result
            final_prompt += f"\nIn the inventory we have {product_name}. The price of this product is {price} dollars. \
                            We have {units} units in stock."
    final_prompt += ")"

    return llm_api(final_prompt)


async def chat():
    st.set_page_config(page_title="Testing AI Assistant")
    st.title('Testing AI Assistant')

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Welcome!"}]

    with st.sidebar:
        if st.button("New Chat"):
            st.session_state.messages = [{"role": "assistant", "content": "Welcome!"}]
            st.session_state.conversation_history = deque(maxlen=5)  # Reset conversation history
            st.session_state.last_mentioned_brand = ""
            st.experimental_rerun()

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_history.append(f"User: {prompt}\n")
        st.chat_message("user").write(prompt)

        if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    try:
                        response = generate_response(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.conversation_history.append(f"Assistant: {response}\n")
                    except Exception as e:
                        st.error("An error occurred, we are working on it")


if __name__ == "__main__":
    asyncio.run(chat())

