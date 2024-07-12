import streamlit as st
import json
import asyncio
import requests
from vector_space_model import handle_query_by_vector_space_model
import ollama

import time

def llm_api(prompt):
    """Call the LLM API with the given prompt and handle exceptions."""
    new_system_prompt = """
    You are a chatbot for a shopping application by the company Jio, called JioMart. Your task is to only respond to user
    queries regarding products in the shop. If the user gives you a greeting, greet them back.
    If the user asks for anything irrelevant or gives a gibberish prompt,
    kindly ask them to ask a question related to the products in JioMart. Your task is only to give current information
    about the products. You cannot add items to cart or give the user notifications. Just answer the questions. 
    """
    try:
        response = ollama.chat(model='llama3', messages=[
            {'role': 'system', 'content': new_system_prompt},
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
    relevant_results = query_data(msg)
    if not relevant_results:
        final_prompt = msg + "\n(We do not have any such items in stock.)"
        return llm_api(final_prompt)

    category_list = ["shoes", "cosmetics", "electronics", "clothing", "accessories", "sports"] # can be read from db
    correct_category = get_category(msg, category_list).lower()

    final_prompt = msg + "(Here is a list of relevant items we have and the info you should use to answer: "

    top = 10
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
            st.experimental_rerun()

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    try:
                        response = generate_response(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error("An error occurred, we are working on it")


if __name__ == "__main__`":
    asyncio.run(chat())