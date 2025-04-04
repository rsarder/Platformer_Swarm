from langchain_openai import ChatOpenAI

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