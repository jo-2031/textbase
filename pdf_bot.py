import logging
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from textbase import bot, Message
from typing import List

# Configure logging
openai_api_key = ""

pdf = "paper27.pdf"

@bot()
def on_message(message_history: List[Message], state: dict = None):
    load_dotenv()
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Validate text extraction result
        if text and text.strip():
            # Split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # Create embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            # Show user input
            user_question = message_history[-1]["content"]

            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                bot_response = chain.run(input_documents=docs, question=user_question)
                print(cb)
            if bot_response:
                response = {
                "data": {
                    "messages": [
                        {
                            "data_type": "STRING",
                            "value": bot_response
                        }
                    ],
                    "state": state
                },
                "errors": [
                    {
                        "message": ""
                    }
                ]
            }

            return {
                "status_code": 200,
                "response": response
            }

