"""
Now that we have knowledge base ready , we can have rag system 
that retrives relevent documents of the similar intent student from 
the knowledge base and generate answers basedon those documents.

we will use groq llm to qna using this vdb

user -> query
query -> embed -> google gemini embedding model -> vector
query vector -> pinecone vdb -> retrieve relevent documents 

Give the relevent documents to the groq llm as an context -> answer

"""

from dotenv import load_dotenv
import os
from pinecone import Pinecone
from create_vectors import embed_text, vector_index
from groq import Groq
import streamlit as st

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


st.title("Rag System for student knowledge base")
st.subheader("Ask me Question about the student")

user_query = st.text_input("Ask me a Question:")

send_btn = st.button("send")

system_prompt = {
    "role": "system",
    "content": """
You are helpful assistant that answer questions about the student.
Anything non relevant to the student should be ignored and politely
say that you don't know the answer.
"""
}

if send_btn and user_query:
    # Embed the user query
    query_vector = embed_text(user_query)

    # Retrieve relevant documents from Pinecone
    response = vector_index.query(
        vector=query_vector,
        top_k=2,  # Retrieve top 2 relevant documents
        include_metadata=True
    )

    # Extract the relevant documents
    similar_documents = ""
    for match in response["matches"]:
        similar_documents += match["metadata"]["text"] + "\n\n"

    # Prepare the context for groq LLM
    user_prompt = {
        "role": "user",
        "content": f"""Here are relevant documents fetched according to 
user question. Use the document to analyze and answer user query:
{similar_documents}
Question: {user_query}"""
    }

    # Generate answer using Groq LLM
    groq_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[system_prompt, user_prompt],
        max_tokens=3000,
        temperature=0.7
    )

    result = groq_response.choices[0].message.content

    st.subheader("Answer")
    st.write(result)
