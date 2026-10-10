"""LLM client using Groq's free tier via LangChain."""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize the chat model with Groq's Gemma2 (free tier)
# Check https://console.groq.com/docs/models for current available models
chat_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY")
)


def ask_llm(prompt: str) -> str:
    """
    Send a prompt to the LLM and return the response.

    Args:
        prompt: The question or instruction to send to the LLM

    Returns:
        The LLM's response as a string
    """
    response = chat_model.invoke(prompt)
    return response.content
