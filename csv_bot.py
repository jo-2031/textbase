from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from typing import List
from textbase import bot, Message

# Define your OpenAI API key
api_key = "Your API key"

@bot()
def on_message(message_history: List[Message], state: dict = None):
    try:
        # Check if there are previous user messages
        if not message_history:
            return {
                "status_code": 200,
                "response": {
                    "data": {
                        "messages": [],
                    },
                }
            }

        # Initialize your CSV agent
        user_csv = "data2.csv"  # Update with the actual path
        agent = create_csv_agent(
            ChatOpenAI(api_key, model="gpt-3.5-turbo-0613"),
            user_csv,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )

        # Extract the user's message from message_history
        user_message = message_history[-1].content

        # Use the user's message as input to the agent
        bot_response = agent.run(user_message)

        response = {
            "data": {
                "messages": bot_response
            },
        }

        return {
            "status_code": 200,
            "response": response
        }
    except Exception as e:
        return {
            "status_code": 500,
            "response": {
                "data": {
                    "messages": [],
                },
                "errors": [{
                    "message": str(e)
                }]
            }
        }
