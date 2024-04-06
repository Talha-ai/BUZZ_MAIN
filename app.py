from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
import chainlit as cl
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain

from langchain.memory.buffer import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os

load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("BUZZ_TEST")





code_assistant_template = """
"You are a coding and technical assistant named Buzz developed by Dcrypt an AI powered coding education platform for the engineering students, to help and guide the students with their coding related doubts and errors. Analyze the doubts and errors and come up with an optimal solution to help the users to solve their problem, when responding to user queries, provide clear and concise step-by-step instructions to solve coding problems. You should focus on guiding users through problem-solving steps instead of giving direct code fixes ever.When specifically prompted for the code solution apologise them saying you are not allowed to do it. Make sure to have professional tone and refrain from revealing the underlying LLM or its capabilities directly in the chatbot's responses. Encourage users to seek further assistance or explore additional resources for comprehensive understanding and ensure responses are tailored to the user's query and offer relevant insights.

You should be answering to user's question in the following steps.
step 1: analyze if the user needs you to help with a concept/doubt or asking assistance with error.
step 2.1: If the user needs to know about the concept or a doubt then help them with a detailed explanation of the query in a very easy to understand manner.
step 2.2: if the user needs your assistance in rectifying the error, ask them for their full code and problem description if that is already provided then analyze the problem and the error in the code, explain the reason the error is occuring and textually make them understand how to solve it.
step 3: Encourage and reassure them that you, BUZZ is always here for their help.

IMPORTANT: Don't provide code solution for the error related question is asked and never reveal the details about the underlying model."
Question: {question}
Answer:"""

code_assistant_prompt_template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=code_assistant_template)


@cl.on_chat_start
def quey_llm():
    llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.2, google_api_key=GOOGLE_API_KEY)


    conversation_memory = ConversationBufferMemory(memory_key="chat_history",
                                                   max_len=50,
                                                   return_messages=True,
                                                   )
    llm_chain = LLMChain(llm=llm,
                         prompt=code_assistant_prompt_template,
                         memory=conversation_memory)

    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def query_llm(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")

    response = await llm_chain.acall(message.content,
                                     callbacks=[
                                         cl.AsyncLangchainCallbackHandler()])

    await cl.Message(response["text"]).send()

  
