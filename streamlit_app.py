import streamlit as st
import uuid
import psycopg2
from config import get_vectorstore_config
import os
import stat
import shutil
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.memory import ConversationSummaryBufferMemory
from langchain_text_splitters import Language
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from urllib.parse import urlparse, urlunparse
from config import  get_vectorstore_config, get_open_ai_api_key
import psycopg2
from pgaiembeddings import PgAIEmbeddings



# Open-Ai Key configuration
vectostore_config = get_vectorstore_config()
open_ai_api_key = st.secrets["open_ai_api_key"]
connection = st.secrets["database_url"]


# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Please provide the answer to the user's questions based on the following content from the provided code files: \n {input_documents} \n as well as use the summary of the previous chat messages between you and the user : \n {chat_summary}.",
        ),
        ("human", "{input}"),
    ]
)


# Helper function for extracting the repository name from repo url
def get_repo_name(repo_url):
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) == 2:
        repo_name = path_parts[1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        return repo_name
    else:
        raise ValueError("Invalid GitHub repository URL")


# Helper function for providing the required permissions for deleting the cloned project directory
def on_rm_error(func, path, exc_info):
    """ Error handler for `shutil.rmtree`. 
        If the error is due to an access error (read-only file), 
        it attempts to add write permission and then retries.
    """
    # Check if the file is read-only
    if not os.access(path, os.W_OK):
        # Add write permission
        os.chmod(path, stat.S_IWUSR)
        # Retry the removal
        func(path)
    else:
        raise


extensions = {"Python":"py", "Java":".java", "JavaScript":".js"}

def get_project_names():
    try:
        conn = psycopg2.connect(connection)
        cursor = conn.cursor()
        # Query to get project names
        cursor.execute("SELECT name FROM public.langchain_pg_collection")
        project_names = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return project_names
    except Exception as e:
        st.error(f"Error fetching project names: {e}")
        return ["No Projects Found!"]

# Sidebar dropdown
st.sidebar.title("Options")
option = st.sidebar.selectbox("Choose an option", ["Process Codebase", "Chatbot"])

# Option 1: Process Codebase
if option == "Process Codebase":
    st.title("Process Codebase 👩‍💻")
    github_url = st.text_input("Enter GitHub Repository URL")
    selected_languages = st.multiselect("Select file types to process:", list(extensions.keys()))

    # Map selected languages to their corresponding extensions
    extension_list = [extensions[lang] for lang in selected_languages]
    # git_token = st.text_input("Enter your person github token")

    if st.button("Process Codebase"):
        if github_url and selected_languages:
            with st.spinner("Processing Codebase.."):
                st.write(f"Processing repository: {github_url}")
                st.write(f"Selected file types: {', '.join(selected_languages)}")

                github_repo_name = get_repo_name(github_url)
                clone_path = os.path.join(".", "temp_cloning_dir", github_repo_name)
                collection_name = github_repo_name

                try:
                    repo = Repo.clone_from(github_url, to_path=clone_path)
                    print("Repository Cloned Successfully!")

                    print(collection_name)
                    print(clone_path)

                    loader = GenericLoader.from_filesystem(
                        clone_path,
                        glob="**/*",
                        suffixes=extension_list,
                        # parser=PyPDFParser()
                        parser=LanguageParser(language=Language.JAVA, parser_threshold=500)
                    )

                    documents = loader.load()
                    print(len(documents))

                    #Splitting of documents
                    text_splitter = RecursiveCharacterTextSplitter.from_language(
                        # Set a really small chunk size, just to show.
                        language=Language.JAVA,
                        chunk_size=2000,
                        chunk_overlap=200,
                    )

                    # print(all_documents)
                    document_chunks = text_splitter.split_documents(documents)
                    print(len(document_chunks))

                    embeddings = OpenAIEmbeddings(api_key=open_ai_api_key)

                    vectorstore = PGVector(
                        embeddings=PgAIEmbeddings(connection_string=connection),
                        collection_name=collection_name,
                        connection=connection,
                        use_jsonb=True,
                    )

                    #Adding documents to vector store
                    vectorstore.add_documents(document_chunks)
                    print("Documents Stored in Vector Store")

                except Exception as e:
                    if 'could not read Password' in str(e):
                        st.error("Authentication failed. Please check your personal access token.")
                    elif 'already exists and is not an empty directory' in str(e):
                        st.error(f"Destination path '{clone_path}' already exists and is not an empty directory. Provide an empty directory path for cloning.")
                    elif 'Repository not found' in str(e):
                        st.error(f"Repository not found. Please check the repository URL: {github_url}")
                    else:
                        st.error(f"Failed to clone repository: {e}")

                finally:
                    # This block always runs, regardless of whether an exception occurred
                    try:
                        shutil.rmtree(clone_path, onerror=on_rm_error)
                        print("Cloned repository deleted successfully.")
                    except OSError as e:
                        st.error(f"Error deleting cloned repository: {e}")
                    st.info("Codebase processed successfully!")

        else:
            st.warning("Please enter a GitHub repo and Select file types.")

# Option 2: Chatbot
elif option == "Chatbot":
    st.title("Chatbot 🤖")
    
    project_names = get_project_names()
    
    if project_names:
        # Custom label styled with a blue background
        st.markdown('<div style="background-color: #e9f5fb; padding: 10px; border-radius: 5px;"><strong style="color: #3178c6;">Select Project</strong></div>', unsafe_allow_html=True)

        # The select box for selecting a project
        project_name = st.selectbox("", project_names)
        
        # Display chat interface
        st.warning("**Chat with Assistant**")
        
        # Initialize chat history in session state if not already set
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            if message["sender"] == "user":
                st.info(f"**You 👤:** {message['content']}")
            else:
                st.success(f"**Assistant 🤖:** {message['content']}")
        
        user_query = st.chat_input("Enter your query", key="chat_input")
        
        if user_query:
            with st.spinner("Generating Response.."):
                embeddings = OpenAIEmbeddings(api_key=open_ai_api_key)

                vectorstore = PGVector(
                    embeddings=embeddings,
                    collection_name=project_name,
                    connection=connection,
                    use_jsonb=True,
                )
                
                found_docs = vectorstore.similarity_search(user_query, k=3)

                llm = ChatOpenAI(api_key=open_ai_api_key)
                parser = StrOutputParser()

                chain = prompt | llm | parser

                if 'chat_messages' not in st.session_state:
                    st.session_state['chat_messages'] = []

                if 'previous_summary' not in st.session_state:
                    st.session_state['previous_summary'] = ""

                # Access chat history and summary from session state
                chat_messages_history = st.session_state['chat_messages']
                previous_summary = st.session_state['previous_summary']

                # Reconstruct the ConversationSummaryBufferMemory object
                memory = ConversationSummaryBufferMemory(llm = llm, max_token_limit=10, return_messages=True)
                memory.chat_memory.messages = chat_messages_history

                output = chain.invoke(
                    {
                        "input_documents": found_docs,
                        "input": user_query,
                        "chat_summary": previous_summary
                    }
                )

                
                memory.chat_memory.add_user_message(user_query)
                memory.chat_memory.add_ai_message(output)
            
                messages = memory.chat_memory.messages
                
                previous_summary = memory.predict_new_summary(messages, previous_summary)
                
                # Update session data
                st.session_state['chat_messages'] = messages
                st.session_state['previous_summary'] = previous_summary

                response = output
                st.write(response)
                # Append user message to chat history
                st.session_state.messages.append({"sender": "user", "content": user_query})
                
                # Call LLM for response (replace with actual LLM call)
                assistant_response = response  # Replace this line with your LLM call
                
                # Append assistant response to chat history
                st.session_state.messages.append({"sender": "assistant", "content": assistant_response})
                
                # Rerun to clear the input field
                st.rerun()

    else:
        st.error("No projects available.")
else:
    st.error("Select an option from the sidebar to begin.")
