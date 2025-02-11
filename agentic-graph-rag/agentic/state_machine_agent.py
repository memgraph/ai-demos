from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import json

# Possible states of finite state machine, used for the agent state management
class State(str):
    INITIALIZING = "INITIALIZING"
    ANALYZING = "ANALYZING"
    SEARCHING = "SEARCHING"
    ANSWERING = "ANSWERING"
    VALIDATING = "VALIDATING"
    COMPLETED = "COMPLETED"

# Define state transition model
class StateTransition(BaseModel):
    current_state: State
    next_states: List[State]
    context: str

# Define state machine model
class FSM(BaseModel):
    state: State
    transitions: Dict[State, StateTransition]
    history: List[State] = Field(default_factory=list)
    state_context: Dict[State, str]

    def update_state(self, new_state: State):
        if new_state in self.transitions[self.state].next_states:
            self.history.append(self.state)
            self.state = new_state
        else:
            raise ValueError(f"Invalid transition from {self.state} to {new_state}")

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

# Example of how to define transitions with context
transitions = {
    State.INITIALIZING: StateTransition(
        current_state=State.INITIALIZING,
        next_states=[State.ANALYZING],
        context="Initializing the system and preparing for analysis."
    ),
    State.ANALYZING: StateTransition(
        current_state=State.ANALYZING,
        next_states=[State.SEARCHING],
        context="Analyzing the input data to determine the next steps."
    ),
    State.SEARCHING: StateTransition(
        current_state=State.SEARCHING,
        next_states=[State.ANSWERING, State.VALIDATING],
        context="Searching for relevant information based on the analysis."
    ),
    State.ANSWERING: StateTransition(
        current_state=State.ANSWERING,
        next_states=[State.COMPLETED],
        context="Providing answers based on the search results."
    ),
    State.VALIDATING: StateTransition(
        current_state=State.VALIDATING,
        next_states=[State.SEARCHING, State.COMPLETED],
        context="Validating the provided answers to ensure accuracy."
    ),
    State.COMPLETED: StateTransition(
        current_state=State.COMPLETED,
        next_states=[],
        context="The process is completed."
    )
}

# State context definitions
state_context = {
    State.INITIALIZING: "Used for setting up the application, and preparing the data.",
    State.ANALYZING: "After INITIALIZING we can move to ANALYZING if the user has provided the question about the GraphRAG dataset. At this point, the agent will analyze the question and based on the information, say what tools and approaches can be used by order. This will depend on the question.",
    State.SEARCHING: "After the analysis, we can call tools by order defined from the analyzing step. After each tool is run, we need to make VALIDATION and get back to searching if validations have failed. As soon as we have a valid answer, we move to the ANSWERING phase.",
    State.ANSWERING: "Here we are passing only the key information to form the best answer.",
    State.COMPLETED: "The application has responded to the user's question."
}

# Initialize the FSM with the initial state, transitions, and state context
FST_manager = FSM(state=State.INITIALIZING, transitions=transitions, state_context=state_context)