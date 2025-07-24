import json
import re
import uuid
from typing import TypedDict, Annotated, Optional, Dict
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, BaseMessage
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_core.agents import AgentAction, AgentFinish
from memory import save_to_db
from agent_tools import tools 
import os
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# State schema
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    thread_id: str
    pcos_prediction: Optional[str]
    prediction_probability: Optional[float]
    features: Dict


llm = ChatGroq(
    model="llama-3.1-8b-instant", temperature=0 
)


def initial_prediction_agent_node(state: AgentState):
    main_messages = []
    agent_scratchpad_messages = []
    
    features_as_json = json.dumps(state["features"])
    patient_data_context_message = HumanMessage(
        content=f"Initial patient data for PCOS prediction is provided below. Please use this data to call the `predict_pcos` tool:\n```json\n{features_as_json}\n```"
    )
    
    initial_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant specialized in PCOS prediction. Your primary function is to call the `predict_pcos` tool based on the user's input.\n"
                "IMPORTANT: You have been provided with patient data in a preceding message as a JSON object.\n"
                "ALWAYS begin by calling the `predict_pcos` tool. Extract individual fields (e.g., 'age', 'weight') from the provided patient data and pass them as precise keyword arguments to the `predict_pcos` tool.\n"
                "Ensure all required parameters for `predict_pcos` are extracted and passed correctly. Do NOT ask for more information.\n"
                "After calling `predict_pcos`, do NOT generate any text. Just provide the tool call."
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent_executor = create_tool_calling_agent(llm, tools, prompt=initial_prompt)
    
    agent_input_messages = [patient_data_context_message] + state["messages"]

    agent_output = agent_executor.invoke(
        {
            "messages": agent_input_messages,
            "intermediate_steps": []  
        },
        config={"recursion_limit": 50}
    )
    
    
    messages_to_add = []
    if isinstance(agent_output, BaseMessage):
        messages_to_add = [agent_output]
    elif isinstance(agent_output, AgentAction): 
        messages_to_add = [AIMessage(content="", tool_calls=[{"name": agent_output.tool, "args": agent_output.tool_input, "id": str(uuid.uuid4())}])]
    elif isinstance(agent_output, list):
        for item in agent_output:
            if isinstance(item, BaseMessage):
                messages_to_add.append(item)
            elif isinstance(item, AgentAction): 
                messages_to_add.append(AIMessage(content="", tool_calls=[{"name": item.tool, "args": item.tool_input, "id": str(uuid.uuid4())}]))
            else:
                print(f"Warning: Unexpected item type in agent_output list: {type(item)}")
    else:
        print(f"Warning: Unexpected agent_output type: {type(agent_output)}")
        

    return {"messages": messages_to_add}


def predict_tool_node(state: AgentState):
    tool_node = ToolNode(tools=tools)
    return tool_node.invoke(state)


def extract_prediction_node(state: AgentState):
    raw_tool_output = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.name == "predict_pcos":
            try:
                raw_tool_output = json.loads(msg.content)
                break
            except json.JSONDecodeError:
                print(f"Error decoding JSON from predict_pcos tool output: {msg.content}")
                continue
    
    if raw_tool_output and "pcos_prediction" in raw_tool_output:
        state["pcos_prediction"] = raw_tool_output["pcos_prediction"]
        state["prediction_probability"] = raw_tool_output.get("prediction_probability")
        state["features"] = raw_tool_output.get("features", state["features"])
        print(f"ExtractPredictionNode: predict_pcos tool returned a string: {raw_tool_output}. Prediction will be {state['pcos_prediction']}.")
    else:
        state["pcos_prediction"] = "Unknown"
        state["prediction_probability"] = None
        print(f"ExtractPredictionNode: Could not find valid prediction in tool output. Output: {raw_tool_output}")
    
    return state


def followup_tool_node(state: AgentState):
    print("Executing Followup Tools directly...")
    pcos_prediction = state.get("pcos_prediction", "Unknown")
    
    features = state.get("features", {}) 
    
    explain_tool = next((t for t in tools if t.name == "explain_pcos_results"), None)
    recommend_tool = next((t for t in tools if t.name == "recommend_next_steps"), None)

    explanation_message = None
    recommendation_message = None
    explanation = ""

    if explain_tool:
        explanation_input = {"pcos_prediction": pcos_prediction, "features": features}
        explanation = explain_tool.invoke(explanation_input)
        explanation_message = ToolMessage(content=explanation, name="explain_pcos_results", tool_call_id=str(uuid.uuid4()))
        print("🔍 explain_pcos_results tool called")
    else:
        explanation_message = ToolMessage(content="Error: explain_pcos_results tool not found.", name="explain_pcos_results", tool_call_id=str(uuid.uuid4()))
        print("Error: explain_pcos_results tool not found.")


    if recommend_tool:
        # Pass the 'explanation' retrieved from the previous tool call
        recommendation_input = {"pcos_prediction": pcos_prediction, "explanation_text": explanation} 
        recommendation = recommend_tool.invoke(recommendation_input)
        recommendation_message = ToolMessage(content=recommendation, name="recommend_next_steps", tool_call_id=str(uuid.uuid4()))
        print("🔍 recommend_next_steps tool called")
    else:
        recommendation_message = ToolMessage(content="Error: recommend_next_steps tool not found.", name="recommend_next_steps", tool_call_id=str(uuid.uuid4()))
        print("Error: recommend_next_steps tool not found.")

    # Return messages only if they were successfully created
    messages_to_return = []
    if explanation_message:
        messages_to_return.append(explanation_message)
    if recommendation_message:
        messages_to_return.append(recommendation_message)
        
    return {"messages": messages_to_return}


def log_node(state: AgentState):
    features_for_db = state.get("features", {}) # This will now be the transformed features
    prediction = state.get("pcos_prediction", "Unknown")
    probability = state.get("prediction_probability", 0.0)

    explanation = "No explanation provided."
    recommendation = "No recommendation provided."

    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            if msg.name == "explain_pcos_results":
                explanation = msg.content
            elif msg.name == "recommend_next_steps":
                recommendation = msg.content

    save_to_db(
        user_input=features_for_db,
        prediction=prediction,
        probability=probability,
        explanation=explanation,
        recommendation=recommendation,
        user_id=state.get("user_id", "anonymous"),
        thread_id=state.get("thread_id", "unknown")
    )
    print("LogNode: Data prepared and saved to DB.")
    return state


# Define the graph
graph = StateGraph(AgentState)

graph.set_entry_point("InitialPredictionAgent")

graph.add_node("InitialPredictionAgent", initial_prediction_agent_node)
graph.add_node("Predict Tool", predict_tool_node)
graph.add_node("Extract Prediction", extract_prediction_node)
graph.add_node("Followup Tools", followup_tool_node)
graph.add_node("Log", log_node)

graph.add_edge("InitialPredictionAgent", "Predict Tool")
graph.add_edge("Predict Tool", "Extract Prediction")
graph.add_edge("Extract Prediction", "Followup Tools")
graph.add_edge("Followup Tools", "Log")
graph.add_edge("Log", END)

app = graph.compile()