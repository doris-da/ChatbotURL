# add customized prompt
# citation ok
# need to work on UI, see test6

import os  
import streamlit as st  
from dotenv import load_dotenv  
from langchain import hub  
from langchain_community.document_loaders import WebBaseLoader  
from langchain_community.vectorstores import FAISS  
from langchain_core.output_parsers import StrOutputParser  
from langchain_core.runnables import RunnablePassthrough  
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings  
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_core.prompts import PromptTemplate  

# Developer-configured websites (modify this list as needed)  
DEVELOPER_URLS = [  
    "https://en.wikipedia.org/wiki/Artificial_intelligence",  
    "https://www.gs.cuhk.edu.hk/admissions/",
    "https://www.hkubs.hku.hk/about-us/overview/message-from-the-dean/" 
]  

# Load environment variables  
load_dotenv()  

def debug_log(message):  
    """Display debug messages in the sidebar."""  
    st.sidebar.code(f"DEBUG: {message}")  

@st.cache_resource  
def initialize_azure_components():  
    """Initialize and cache Azure LLM and embeddings components."""  
    try:  
        #debug_log("Initializing Azure components...")  
        embeddings = AzureOpenAIEmbeddings(  
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),  
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")  
        )  
        llm = AzureChatOpenAI(  
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),  
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),  
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
            temperature=0.5  
        )  
        #debug_log("Azure components initialized successfully.")  
        return embeddings, llm  
    except Exception as e:  
        st.error(f"Azure initialization failed: {str(e)}")  
        st.stop()  

@st.cache_resource  
def initialize_vectorstore(_embeddings):  
    """Initialize and cache vector store from developer-configured URLs."""  
    try:  
        #debug_log("Starting web content processing...")  
        
        # Load web content from developer URLs  
        loader = WebBaseLoader(DEVELOPER_URLS)  
        docs = loader.load()  
        
        if not docs:  
            raise ValueError("No readable content found on the provided websites")  

        #debug_log(f"Loaded {len(docs)} document pages from websites.")  

        text_splitter = RecursiveCharacterTextSplitter(  
            chunk_size=800, chunk_overlap=240  
        )  
        splits = text_splitter.split_documents(docs)  
        if not splits:  
            raise ValueError("Text splitting resulted in zero chunks.")  

        #debug_log(f"Created {len(splits)} text chunks.")  

        vector_db = FAISS.from_documents(documents=splits, embedding=_embeddings)  
        #debug_log("Vector store initialized successfully.")  
        return vector_db, docs  
    except Exception as e:  
        st.error(f"Vector store error: {str(e)}")  
        st.stop()  

# Initialize session state  
if "agreed_to_disclaimer" not in st.session_state:  
    st.session_state.agreed_to_disclaimer = False  

if "messages" not in st.session_state:  
    st.session_state.messages = []  

# Main application logic  
if not st.session_state.agreed_to_disclaimer:  
    with st.container():  
        st.markdown("""  
        <div style='text-align: center; max-width: 600px; margin: 0 auto;'>  
            <h1 style='color: #2A5C84; margin-bottom: 30px;'>Disclaimer</h1>  
            <p style='margin-bottom: 30px;'>This Chatbot is powered by the GPT-4o AI language model, developed by OpenAI, and is intended strictly for informational purposes. It is specifically designed to address inquiries related to the services and offerings of HKU Business School Masters Programs. While every effort has been made to ensure the accuracy and relevance of the information provided, please note that the Chatbot's responses may not always reflect the most recent updates or developments and cannot be guaranteed to be completely accurate at all times.</p>  
        </div>  
        """, unsafe_allow_html=True)  
        
        col1, col2, col3 = st.columns([1, 2, 1])  
        with col2:  
            button_style = """  
            <style>  
                .stButton > button {  
                    background-color: #4CAF50 !important;  
                    color: white !important;  
                    border: none;  
                    padding: 15px 32px;  
                    text-align: center;  
                    text-decoration: none;  
                    display: inline-block;  
                    font-size: 16px;  
                    margin: 4px 2px;  
                    cursor: pointer;  
                    border-radius: 12px;  
                    width: 100%;  
                }  
                .stButton > button:hover {  
                    background-color: #45a049 !important;  
                }  
            </style>  
            """  
            st.markdown(button_style, unsafe_allow_html=True)  
            
            if st.button("I Agree", key="agree_button"):  
                st.session_state.agreed_to_disclaimer = True  
                st.rerun()  
else:  
    with st.sidebar:  
        st.header("Diagnostics")  
        if st.button("Clear Cache"):  
            st.cache_resource.clear()  
            st.session_state.agreed_to_disclaimer = False  

    try:  
        embeddings, llm = initialize_azure_components()  
        vector_db, docs = initialize_vectorstore(embeddings)  
        retriever = vector_db.as_retriever()  
    except Exception as e:  
        st.error(f"Failed to initialize essential components: {e}")  
        st.stop()  
    
  
    st.markdown("""  
    <h1 style='text-align: center; color: #2A5C84; font-family: Arial, sans-serif; margin-bottom: 20px;'>  
    🤖HKU Business School Chatbot   
    <span style='color: #DAA520; font-size: 0.4em;'>Supported by Azure OpenAI</span>  
    </h1>  
    <hr style='border: 1.5px solid #2A5C84; margin: 10px auto; width: 90%;'>  
    """, unsafe_allow_html=True)  

    st.write(f"Configured knowledge base: {len(DEVELOPER_URLS)} official website(s)")  


    # Display chat messages from history on app rerun  
    # Display chat history first  
    for message in st.session_state.messages:  
        with st.chat_message(message["role"]):  
            st.markdown(message["content"], unsafe_allow_html=True)  
            if message["role"] == "assistant" and message.get("html_refs"):  
                st.markdown(message["html_refs"], unsafe_allow_html=True)  
    
    if prompt := st.chat_input("Type a NEW question about HKU Business School Masters Programs"):  
        st.session_state.messages.append({"role": "user", "content": prompt})  

        with st.chat_message("user"):  
            st.markdown(prompt)  

        with st.spinner("Exploring the info to make things crystal clear!"):  
            try:  
                # Retrieve documents based on the prompt  
                documents = retriever.get_relevant_documents(prompt)  

                if not documents:  
                    response = "I don't know."  
                    formatted_refs = ""  
                else:  
                    # Construct context from retrieved documents  
                    context = "\n\n".join([  
                        f"- {doc.page_content}\n(Source: {doc.metadata['source']})"  
                        for doc in documents  
                    ])  

                    # Create a single comprehensive prompt template   

                    custom_rag_prompt = PromptTemplate.from_template("""  
                    You are an intelligent chatbot designed to answer questions only based on the provided documents.
                    The context contains content from specific authoritative websites along with their source URLs.  
                                                                     
                    Follow these rules:  
                    - Reference only the information found in the provided context.                                                 
                    - If the input is a casual greeting (e.g., "Hi," "Hello"), respond warmly without adding assumptions.  
                    - If the input is unclear, politely ask for clarification.  
                    - Do not make up any information or assumptions.
                    - When possible, mention which source website the information comes from.   
                    - If the answer to a question cannot be found in these documents, or if you are unsure, reply with "I don't know" and do not make up any information or assumptions.  
                    - Always prioritize accuracy and refer only to the provided website content for your answers. 
                    - Use bullet points or multiple paragraphs for clarity.  
                    - Only when your response contains factual information from the provided context, end with: "Please visit the official website to learn more about your future at HKU Business School."  

                    Context: {context}  

                    Question: {question}  

                    Answer:  
                    """)  

                    # Prepare the input for the chain  
                    chain_input = {  
                        "context": context,  
                        "question": prompt  
                    }  

                    # Invoke the chain  
                    rag_chain = (  
                        RunnablePassthrough() | custom_rag_prompt | llm | StrOutputParser()  
                    )  
                    response = rag_chain.invoke(chain_input)  

                                        
                    # Handle unknown responses  
                    unknown_phrases = [  
                        "i don't know",  
                        "no information",  
                        "not contain information",  
                        "does not mention",  
                        "no relevant sources"  
                    ]  

                    if any(phrase in response.lower() for phrase in unknown_phrases):  
                        response = "I apologize, but I don't have sufficient information to answer that question. For detailed inquiries, please contact the admissions office directly at masters@hku.hk."  
                        unique_sources = []  
                    else:  
                        # Extract sources if there are relevant matches  
                        relevant_sources = []  
                        for doc in documents:  
                            if any(keyword in response.lower() for keyword in doc.page_content.lower().split()):  
                                relevant_sources.append(doc.metadata["source"])  

                        # Deduplicate the sources  
                        seen = set()  
                        unique_sources = []  
                        for url in relevant_sources:  
                            if url not in seen:  
                                seen.add(url)  
                                unique_sources.append(url)  

                    # Format references separately  
                    formatted_refs = ""  
                    if unique_sources:  
                        ref_list = [f"[{i}] {url}" for i, url in enumerate(unique_sources, 1)]  
                        formatted_refs = "<br>References:<br>" + "<br>".join(ref_list)  

            except Exception as e:  
                response = f"Information retrieval error: {str(e)}"  
                formatted_refs = ""  

        # Store response and references separately  
        st.session_state.messages.append({  
            "role": "assistant",  
            "content": response,  
            "html_refs": formatted_refs  
        })  

        st.rerun()  