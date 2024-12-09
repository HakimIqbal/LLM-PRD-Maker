import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from typing import Dict

# Load environment variables
load_dotenv()

# Dictionary untuk menyimpan memory untuk setiap session
conversation_memories: Dict[str, ConversationBufferMemory] = {}

# Initialize Langsmith client and tracer
client = Client()
tracer = LangChainTracer(project_name=os.getenv("LANGCHAIN_PROJECT"))

# Setup callback manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler(), tracer])

# Initialize OpenAI LLM with LangChain
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4",  # Bisa disesuaikan, seperti "gpt-3.5-turbo" jika diinginkan
    temperature=0.7,
    streaming=True,
    callback_manager=callback_manager
)


def get_or_create_memory(session_id: str) -> ConversationBufferMemory:
    """Get or create memory for a session."""
    if session_id not in conversation_memories:
        # Tidak menyimpan chat_history, hanya menyimpan human input
        conversation_memories[session_id] = ConversationBufferMemory(
            memory_key="human_input",
            input_key="human_input"
        )
    return conversation_memories[session_id]

def create_chain(prompt_template, memory):
    """Create a chain using LLMChain."""
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    return llm_chain

def load_prompt_from_file():
    """Load the prompt template from a file."""
    try:
        with open("prompts.txt", "r") as file:
            prompt_text = file.read().strip()

            # Menambahkan instruksi tambahan
            personas = """
            Ensure that the generated output strictly adheres to the context of the provided overview.
            Do not include information or details that are unrelated to or extend beyond the overview.
            """

            full_prompt = personas + "\n\n" + prompt_text

            return PromptTemplate(
                input_variables=[
                    "overview", "start_date", "end_date", "document_version",
                    "product_name", "document_owner", "developer",
                    "stakeholder", "doc_stage", "created_date"
                ],
                template=full_prompt
            )
    except FileNotFoundError:
        print("Error: File 'prompts.txt' not found.")
        return None