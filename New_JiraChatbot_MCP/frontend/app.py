import streamlit as st
import httpx

st.title("Jira Metrics Chatbot")

st.write("Enter your Jira question in natural language (e.g., 'How many bugs are there in the sprint?'):")

query = st.text_input("Your question:")

if st.button("Ask Jira") and query:
    with st.spinner("Querying Jira..."):
        try:
            response = httpx.post(
                "http://localhost:8000/query",
                json={"query": query},
                timeout=10
            )
            if response.status_code == 200:
                answer = response.json().get("answer", "No answer returned.")
                st.success(answer)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}") 