from techies.fixture_loader import load_fixture
from techies.tools import get_all_tools
from crewai import Agent as _Agent
from agentops import track_agent

from langchain_openai import ChatOpenAI
import agentops
from agentops import track_agent

class ChatOpenAINoTemp(ChatOpenAI):
    def __init__(self, model_name: str = "o1-2024-12-17", **kwargs):
        # Remove temperature if present
        kwargs.pop("temperature", None)
        super().__init__(model_name=model_name, **kwargs)

    @property
    def _default_params(self) -> dict:
        # Remove temperature from the default params
        params = super()._default_params
        params.pop("temperature", None)
        return params
    
@track_agent()
class Agent(_Agent):  # Inherit from _Agent class
    @staticmethod
    def eager_load_all(**extra_kwargs):
        # Initialize an empty dictionary to store agent instances
        agent_pool = {}
        
        # Retrieve all available tools
        all_tools = get_all_tools()
        
        # Load agent configurations from a fixture (assumed to be a stored config file or dictionary)
        for config_name in load_fixture('agents').keys():
            # Skip any configuration names that start with an underscore (likely reserved configs)
            if not config_name.startswith('_'):
                # Create an agent instance with the specified configuration
                agent = Agent(
                    config_name,
                    agent_pool=agent_pool,  # Pass the agent pool for tracking agents
                    tools_available=all_tools,  # Provide the available tools
                    **extra_kwargs  # Include any additional keyword arguments
                )
        
        # Return the dictionary containing all agent instances
        return agent_pool
    
    def __init__(
        self, config_name, *, agent_pool=None, tools_available=None, llm = None, **kwargs
    ):
        # Load the agent's configuration details from the fixture
        agent_config = load_fixture('agents')[config_name]
        
        # Assign the role and name based on the configuration name
        agent_config['role'] = config_name
        agent_config['name'] = config_name.replace('_', ' ').title()  # Convert underscores to spaces and capitalize
        
        # Initialize an empty list to store the agent's tools
        agent_tools = []
        
        # Iterate through the agent's tool names (if any) and retrieve them from the available tools
        for tool_name in agent_config.get('tools', []):
            agent_tools.append(tools_available.get(tool_name))  # Raise an error if the tool is not available
        
        # Update the agent configuration with the retrieved tools
        agent_config['tools'] = agent_tools


        if agent_config['llm'] == ['o1-2024-12-17']:
            agent_config['llm'] = ChatOpenAINoTemp(model="o1-2024-12-17")
        else: 
            agent_config['llm'] = ChatOpenAI(model=agent_config['llm'][0], temperature=0.2)

        
        # If an agent pool is provided, store the newly created agent in the pool
        if agent_pool is not None:
            agent_pool[config_name] = self
        
        # Update agent_config with any additional keyword arguments
        agent_config.update(kwargs)
        
        # Call the parent class's constructor with the updated configuration
        super().__init__(**agent_config)