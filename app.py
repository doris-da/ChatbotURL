# go to top works, at the left bottom corner
# go to top buttons is in the same container as the input box
# clear chat button, move it to bottom 
# chatbox has border with height 200
# sample questions multiple rows, done
# is working on automaticcly locate to the start of the new question
# can locate to the new question and response, but can;t show the question input box and buttons
# expand and collapse sample questions, after clicking the sample question, it will collapse the sample questions container
# but has warning --> remove all wanrings from streamlit (not done yet)
# done on Modify the Website Loader to Parse Hyperlinks and Display them in the Response 
# dealing with the delay between default avatar and custom avator --> test8.py



import streamlit as st  
import os  
from streamlit_scroll_to_top import scroll_to_here # Import the library  
from dotenv import load_dotenv  
from langchain import hub  
from langchain_community.document_loaders import WebBaseLoader  
from langchain_community.vectorstores import FAISS  
from langchain_core.output_parsers import StrOutputParser  
from langchain_core.runnables import RunnablePassthrough  
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings  
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_core.prompts import PromptTemplate  
from functools import partial # Import partial for button callbacks  
from bs4 import BeautifulSoup  
import requests  
from langchain.schema import Document  

# Set page title and chatbot icon  
st.set_page_config(  
    page_title="HKU Business School Chatbot",  
    page_icon="C:/Users/Doris_Jiao/OneDrive - The University Of Hong Kong/Desktop/Revelio/python code/chatpdf/chatbot_icon3.png"  # Replace with the actual path to your icon  
)  


# Developer-configured websites (modify this list as needed)  
DEVELOPER_URLS = [  
    "https://www.hkubs.hku.hk/faqs-for-msc-students",  
    "https://masters.hkubs.hku.hk/articles/masterofaccounting",  
    "https://masters.hkubs.hku.hk/articles/masterofaccounting/wordsfromprogrammedirector"  
]  

# --- Define Sample Questions ---  
SAMPLE_QUESTIONS = [  
    "How many modules are there in total?",  
    "When and where should I apply for the Hong Kong Identity Card?",  
    "Who is the program director for MAcc?",
    "What are some common questions asked by students?",
    "How to apply for student visa?",
    "Why should I choose HKU Business School for my master's program?",
    "Lost & Found Arrangements?",
    "Can I choose to have double majors, and when to do so?",
    "What is the main communication channel of the student and the school?"
]  
# --- End Sample Questions ---  


# Load environment variables  
load_dotenv()  

def debug_log(message):  
    """Display debug messages in the terminal."""  
    print(f"DEBUG: {message}")  

@st.cache_resource  
def initialize_azure_components():  
    """Initialize and cache Azure LLM and embeddings components."""  
    try:  
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
        debug_log("Azure components initialized successfully.")  
        return embeddings, llm  
    except Exception as e:  
        st.error(f"Azure initialization failed: {str(e)}")  
        st.stop()  


class CustomWebBaseLoader:  
    """  
    Custom WebBaseLoader that extracts content from web pages while preserving hyperlinks.  
    """  
    def __init__(self, urls):  
        self.urls = urls  

    def load(self):  
        documents = []  
        for url in self.urls:  
            try:  
                # Fetch the web page content  
                response = requests.get(url)  
                response.raise_for_status()  # Raise an error for bad status codes  

                # Parse the HTML content  
                soup = BeautifulSoup(response.text, 'html.parser')  

                # Process text and embed hyperlinks  
                for anchor in soup.find_all('a', href=True):  
                    # Embed URL as part of the anchor text  
                    # Example: convert <a href="https://...">here</a> to "here [Link: https://...]"  
                    anchor.insert_after(f" [Link: {anchor['href']}]")  

                # Extract full text content (including appended links) from the HTML  
                page_content = soup.get_text()  

                # Append to the list of documents, preserving metadata such as the source URL  
                documents.append(  
                    Document(page_content=page_content, metadata={"source": url})  
                )  

            except Exception as e:  
                debug_log(f"Error loading {url}: {str(e)}")  # Log any errors  
                continue  

        return documents  
    

@st.cache_resource  
def initialize_vectorstore(_embeddings):  
    """Initialize and cache vector store from developer-configured URLs."""  
    try:  
        debug_log("Starting web content processing...")  

        # Load web content from developer URLs  
        loader = CustomWebBaseLoader(DEVELOPER_URLS)  
        docs = loader.load()  

        if not docs:  
            raise ValueError("No readable content found on the provided websites")  

        debug_log(f"Loaded {len(docs)} document pages from websites.")  

        text_splitter = RecursiveCharacterTextSplitter(  
            chunk_size=800, chunk_overlap=240  
        )  
        splits = text_splitter.split_documents(docs)  
        if not splits:  
            raise ValueError("Text splitting resulted in zero chunks.")  

        debug_log(f"Created {len(splits)} text chunks.")  

        vector_db = FAISS.from_documents(documents=splits, embedding=_embeddings)  
        debug_log("Vector store initialized successfully.")  
        return vector_db, docs  
    except Exception as e:  
        st.error(f"Vector store error: {str(e)}")  
        st.stop()  

# Initialize session state  
if "agreed_to_disclaimer" not in st.session_state:  
    st.session_state.agreed_to_disclaimer = False  

if "messages" not in st.session_state:  
    st.session_state.messages = []  

if 'scroll_to_top' not in st.session_state:  
    st.session_state.scroll_to_top = False  

# --- Session State for Expand/Collapse ---  
if "expand_questions" not in st.session_state:  
    st.session_state.expand_questions = False  # Start with the container expanded 

# --- Session State for Sample Question Click ---  
if 'clicked_sample_question' not in st.session_state:  
    st.session_state.clicked_sample_question = None  
# --- End Session State for Sample Question Click ---  



# --- Function Definitions ---  
def scroll():  
    """Sets the session state flag to trigger scrolling."""  
    st.session_state.scroll_to_top = True  

def clear_chat():  
    """Clears the chat history."""  
    st.session_state.messages = []  
    # Also clear any pending clicked question  
    st.session_state.clicked_sample_question = None  


def handle_sample_question_click(question):  
    """Sets the clicked sample question in session state and collapses the expander."""  
    st.session_state.clicked_sample_question = question  
    st.session_state.expand_questions = False  # Collapse the expander  
    st.rerun()  # Force UI update 

# --- End Function Definitions ---  

# Main application logic  
if not st.session_state.agreed_to_disclaimer:  
    # --- Disclaimer Screen (unchanged) ---  
    with st.container():  
        st.markdown("""  
        <div style='text-align: center; max-width: 600px; margin: 0 auto;'>  
            <h1 style='color: #2A5C84; margin-bottom: 30px;'>Disclaimer</h1>  
            <p style='margin-bottom: 30px;'>This Chatbot is powered by the GPT-4o AI language model, developed by OpenAI, and is intended strictly for informational purposes. It is specifically designed to address inquiries related to the services and offerings of HKU Business School Masters Programs. While every effort has been made to ensure the accuracy and relevance of the information provided, please note that the Chatbot's responses may not always reflect the most recent updates or developments and cannot be guaranteed to be completely accurate at all times.</p>  
        </div>  
        """, unsafe_allow_html=True)  
        col1, col2, col3 = st.columns([1, 2, 1])  
        with col2:  
            button_style = """<style>.stButton > button {background-color: #4CAF50 !important;color: white !important;border: none;padding: 15px 32px;text-align: center;text-decoration: none;display: inline-block;font-size: 16px;margin: 4px 2px;cursor: pointer;border-radius: 12px;width: 100%;}.stButton > button:hover {background-color: #45a049 !important;}</style>"""  
            st.markdown(button_style, unsafe_allow_html=True)  
            if st.button("I Agree", key="agree_button"):  
                st.session_state.agreed_to_disclaimer = True  
                st.rerun()  
    # --- End Disclaimer Screen ---  

else:  
    # --- Scroll Execution Logic ---  
    if st.session_state.scroll_to_top:  
        scroll_to_here(0, key='top')  
        st.session_state.scroll_to_top = False  
    # --- End Scroll Execution Logic ---  

    try:  
        embeddings, llm = initialize_azure_components()  
        vector_db, docs = initialize_vectorstore(embeddings)  
        retriever = vector_db.as_retriever()  
    except Exception as e:  
        st.error(f"Failed to initialize essential components: {e}")  
        st.stop()  

    # --- Define the scroll target location ---  
    scroll_to_here(0, key='top')  
    # --- End Scroll Target ---  

    # --- Title ---  
    st.markdown("""  
    <h1 style='text-align: center; color: #2A5C84; font-family: Arial, sans-serif; margin-bottom: 20px;'>  
    🤖HKU Business School Chatbot  
    <span style='color: #DAA520; font-size: 0.4em;'>Supported by Azure OpenAI</span>  
    </h1>  
    """, unsafe_allow_html=True)  
    # --- End Title ---  

    # --- Display Sample Questions ---  
    st.markdown("<h5 style='text-align: center; color: #555; margin-bottom: 10px;'>Explore sample questions:</h5>", unsafe_allow_html=True)  


    with st.expander("Click to view sample questions", expanded=st.session_state.expand_questions):  
        # Break sample questions into multiple rows for better layout  
        rows = (len(SAMPLE_QUESTIONS) + 2) // 3  # Makes 3 questions per row (-ish)  
        for row in range(rows):  
            cols = st.columns(3)  # 3 questions per row  
            for i, question in enumerate(SAMPLE_QUESTIONS[row * 3:(row + 1) * 3]):  # Slice 3 per row  
                with cols[i]:  
                    st.button(  
                        question,  
                        on_click=partial(handle_sample_question_click, question),  
                        key=f"sample_{row}_{i}",  
                        use_container_width=True,  
                        help="Click to ask this question!"  
                    )



    # Create a container for chat messages  
    chat_container = st.container(height=500) # Adjust height as needed  


    # #Display chat messages from history on app rerun  
    # with chat_container:  
    #     for message in st.session_state.messages:  
    #         with st.chat_message(message["role"]):  
    #             st.markdown(message["content"], unsafe_allow_html=True)  
    #             if message["role"] == "assistant" and message.get("html_refs"):  
    #                 st.markdown(message["html_refs"], unsafe_allow_html=True)  
    
    # Set custom user and assistant avatars as global variables for consistent use  
    USER_ICON = "C:/Users/Doris_Jiao/OneDrive - The University Of Hong Kong/Desktop/Revelio/python code/chatpdf/chatbot_icon4.png"  
    ASSISTANT_ICON = "C:/Users/Doris_Jiao/OneDrive - The University Of Hong Kong/Desktop/Revelio/python code/chatpdf/chatbot_icon3.png"  

    # Display chat messages from history on app rerun  
    with chat_container:  
        for message in st.session_state.messages:  
            if message["role"] == "user":  
                with st.chat_message("user", avatar=USER_ICON):  
                    st.markdown(message["content"], unsafe_allow_html=True)  

            elif message["role"] == "assistant":  
                with st.chat_message("assistant", avatar=ASSISTANT_ICON):  
                    st.markdown(message["content"], unsafe_allow_html=True)  

                    # Display references if available  
                    if message.get("html_refs"):  
                        st.markdown(message["html_refs"], unsafe_allow_html=True)


    # --- Container for bottom elements (Buttons + Input) ---  
    bottom_container = st.container()  
    with bottom_container:  
        # Use columns to place buttons side-by-side  
        col_btn1, col_btn2 = st.columns([1,1]) # Adjust ratios if needed  
        with col_btn1:  
             st.button("⬆️ Scroll to Top", on_click=scroll, use_container_width=True)  
        with col_btn2:  
            # Pass the function directly to on_click  
            st.button("🧹 Clear Chat", on_click=clear_chat, help="Reset conversation history", use_container_width=True)  

        # Input box (remains at the bottom conceptually, below the buttons)  
        prompt_from_input = st.chat_input("Type a NEW question about HKU Business School Masters Programs")  
    # --- End Bottom Container ---  

    # --- Determine if a question needs processing ---  
    # Priority: 1. Clicked Sample Question, 2. Text Input  
    triggered_question = None  
    if st.session_state.clicked_sample_question:  
        triggered_question = st.session_state.clicked_sample_question  
        # Add the clicked question to messages *now* before processing  
        st.session_state.messages.append({"role": "user", "content": triggered_question})  
        st.session_state.clicked_sample_question = None # Reset after capturing  
    elif prompt_from_input:  
        triggered_question = prompt_from_input  
        # Add the typed question to messages  
        st.session_state.messages.append({"role": "user", "content": triggered_question})  
    # --- End Determining Question ---  


    # --- Process the triggered question (if any) ---  
    if triggered_question:  
        # Display the user message that triggered this processing run  
        # (This might seem redundant if coming from input, but ensures consistency  
        # and displays the sample question correctly when clicked)  
        with chat_container:  
             with st.chat_message("user",avatar=USER_ICON):  
                st.markdown(triggered_question) # Display the actual question being processed  

        # Generate and display the assistant's response  
        with chat_container:  
            with st.spinner("Exploring the info to make things crystal clear!"):  
                try:  
                    # Retrieve documents based on the triggered question  
                    documents = retriever.get_relevant_documents(triggered_question)  

                    if not documents:  
                        response = "I don't know."  
                        formatted_refs = ""  
                    else:  
                        context = "\n\n".join([  
                            f"- {doc.page_content}\n(Source: {doc.metadata['source']})"  
                            for doc in documents  
                        ])  

                        custom_rag_prompt = PromptTemplate.from_template("""  
                        You are an intelligent chatbot designed to answer questions only based on the provided documents.  
                        The context contains content from specific authoritative websites along with their source URLs.  

                        Follow these rules:  
                        - Reference only the information found in the provided context.  
                        - If the input is a casual greeting (e.g., "Hi," "Hello"), respond warmly without adding assumptions.  
                        - If the input is unclear, politely ask for clarification.  
                        - Do not make up any information or assumptions.  
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
                            "question": triggered_question # Use the triggered question  
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
                                # Consider refining source relevance logic if needed  
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

            # Store assistant response BEFORE the rerun  
            st.session_state.messages.append({  
                "role": "assistant",  
                "content": response,  
                "html_refs": formatted_refs  
            })  

            # Rerun to display the latest full message list  
            st.rerun()  
    # --- End Process triggered question ---  
