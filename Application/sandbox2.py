from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
from Application.Chroma_db import save_data, query_data
from langchain.tools import tool
@tool
def search_vector_db(text: str):
    """if user asks something related to personal info, Search Vector DB for user details based on the given query text.Returns a list of related documents that match the query text to use as context.

    Args:
        text (str): The information query text required, to search for.
        
    Returns:
        documents (list): A list of related documents that match the query text to use as context.
    """
    print('search_vector_db:',text)
    res = query_data(query=text)
    print(res)
    return str(res)

@tool
def store_data(user_info: dict):
    """extract user info and provide as `user_info` argument to this function. Stores any new personal information (like name, location, interests, job, etc.) in a vector database.

    Args:
        user_info (dict): A dictionary containing one or more user's personal information as summarized as key-value pair.
        example: {"name": "John Doe", "location": "New York", "interests": ["reading", "hiking"]} or {"name": "John Doe"} only would be acceptible too.

    Returns:
        str: A message indicating whether the personal information was stored successfully.
    """
    print('store_data:',user_info)

    similar_data = query_data(query=str(user_info))
    print('similar_data:',similar_data)

    for key in user_info.keys():
        if similar_data:
            for data in similar_data:
                print(eval(data['data']))

                if key in eval(data['data']):
                    print('get id')
                    print('update key:value with id')
                    break
            else:
                print('save key:value')
        else:
            # save_data(data=user_info)
            print('save key:value')


    if not user_info:
        return "No personal data to store."

    print(f"Storing personal data: {user_info}")

    return f"✅ Personal info stored successfully: {user_info}"

tool = TavilySearchResults(max_results=2)
# res = tool.invoke({"query": "Where is the Eiffel Tower?"})
# print(res)
tools = [tool, search_vector_db, store_data]
# llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    # graph.update_state(config, {"messages": [llm_with_tools.invoke(state["messages"])]})
    print('state:',state)
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool, search_vector_db, store_data])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    print('start')
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]},config):
        print('-----------------------------------------------------------------')
        print('event:',event)
        for value in event.values():
            # print(value['messages'])
            print('-----------------------------------------------------------------')
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break