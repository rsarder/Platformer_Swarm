import json
import os
import re
import openai
import requests

import numpy as np
import faiss

from bs4 import BeautifulSoup
from crewai.tools import BaseTool
from freesound import FreesoundClient
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from tempfile import TemporaryDirectory
from typing import Type, List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from langchain_community.utilities import GoogleSerperAPIWrapper
from openai import OpenAI
from crewai_tools import DallETool


class ReadFileToolSchema(BaseModel):
    path: str = Field(type=str, description="The path to the file to read.")


class BatchReadFilesToolSchema(BaseModel):
    paths: List[str] = Field(
        type=List[str], description="An array of filenames to read."
    )


class WriteFileToolSchema(BaseModel):
    path: str = Field(type=str, description="The path to the file to write.")

    content: str = Field(
        type=str,
        description=
        "The content to write to the file. Field should be formatted as a string."
    )

class ListFilesToolSchema(BaseModel):
    pass

class SearchSoundToolSchema(BaseModel):
    query: str = Field(type=str, description="Search query for the sound.")
    min_duration: int = Field(
        type=int, description="Minimum duration of the sound in seconds."
    )
    max_duration: int = Field(
        type=int, description="Maximum duration of the sound in seconds."
    )
    max_results: int = Field(
        default=8,
        type=int,
        description="Maximum number of search results to return."
    )

class SaveSoundToolSchema(BaseModel):
    sound_id: int = Field(type=int, description="ID of the sound to save.")
    file_name: str = Field(
        type=str, description="Name of the saved sound file."
    )

class ReadHtmlExamplesToolSchema(BaseModel):
    pass

class QueryMechanicsToolSchema(BaseModel):
    
    query: str = Field(type=str, description="Search query for game mechanic.")

class GoogleSearchToolSchema(BaseModel):
    query: str = Field(type=str, description="The search query to use for Google search.")

class ReadScaffoldToolSchema(BaseModel):
    mode: Optional[str] = Field(
        default="read",
        description="Operation mode: 'list' to list available scaffold files, or 'read' to read file content."
    )
    filename: Optional[str] = Field(
        default=None,
        description="If mode is 'read', specify the name of the file to view. Use 'all' to view all files."
    )
    
class SearchAndSaveSoundToolSchema(BaseModel):
    """
    Defines the input arguments needed to perform a Freesound search
    and save the first matching sound.
    """
    description: str = Field(..., description="Formatted query for Freesound API. For every term, you can use '+' and '-' modifier characters to indicate that a term is 'mandatory' or 'prohibited' (by default, terms are considered to be 'mandatory'). For example, in a query such as query=term_a -term_b, sounds including term_b will not match the search criteria. You are encouraged to generate a formatted query based on your sound description. Avoid using the word sound in the query unless necessary.")
    output_path: str = Field(..., description="Local file path where the chosen sound will be saved.")
    min_duration: int = Field(5, description="Minimum sound duration (seconds).")
    max_results: int = Field(8, description="Maximum number of results to fetch.")

class GenerateAndDownloadImageSchema(BaseModel):
    """
    Schema defining the arguments for generating and downloading an image.
    """
    prompt: str = Field(..., description="Text prompt describing the image you want DALL·E to create.")
    file_name: str = Field(..., description="Local file name (or path) where the generated image is saved.")
    size: str = Field("...", description="Resolution of the generated image, e.g. '1024x1024'.")
    response_format: str = Field("url", description="Either 'url' or 'b64_json'. 'url' is default.")
    # If you want to specify a particular model (e.g., 'dall-e-3'), add:
    # model: str = Field("image-alpha-001", description="The DALL·E model name if needed.")

class ReadFileTool(BaseTool):
    name: str = "Read a File"
    id: str = "read_file"
    description: str = "Read the contents of a file from the bucket."
    args_schema: Type[BaseModel] = ReadFileToolSchema
    base_dir: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, **kwargs) -> str:
        try:
            path = kwargs['path']
            with open(f"{self.base_dir}/{path}", "r") as f:
                content = f.read()

            return content
        except Exception as e:
            files_available = "\t".join(os.listdir(self.base_dir))
            return f"Failed to read file: {e}.\nFiles available: {files_available}"


class BatchReadFilesTool(BaseTool):
    name: str = "Read Some Files"
    id: str = "batch_read_files"
    description: str = "Read contents of files. Same as read_file but for multiple files."
    args_schema: Type[BaseModel] = BatchReadFilesToolSchema
    base_dir: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, **kwargs) -> str:
        try:
            paths = kwargs['paths']
            content = ""
            for path in paths:
                with open(f"{self.base_dir}/{path}", "r") as f:
                    content += f.read()
                    content += "\n"

            return content
        except Exception as e:
            return f"Failed to read files: {e}"


class WriteFileTool(BaseTool):
    name: str = "Update a File"
    id: str = "write_file"
    description: str = "Write content of a file to the bucket."
    args_schema: Type[BaseModel] = WriteFileToolSchema
    base_dir: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, **kwargs) -> str:
        try:
            import difflib

            path = kwargs['path'].replace("/", "-")
            content = kwargs['content']

            if not isinstance(content, str):
                content = json.dumps(content)

            if not os.path.exists(self.base_dir):
                with open(f"{self.base_dir}/{path}", "w") as f:
                    f.write(content)

                return f"File {path} created successfully."
            elif not os.path.exists(f"{self.base_dir}/{path}"):
                with open(f"{self.base_dir}/{path}", "w") as f:
                    f.write(content)
                return f"File {path} created successfully."
            else:
                with open(f"{self.base_dir}/{path}", "r") as f:
                    old_content = f.read()

                if path.endswith(".html") and (len(old_content.splitlines()) * 0.8 > len(content.splitlines())):
                    return f"""
Updating file {path} is rejected because new content is significantly smaller than the old content.
You should include every unchanged line in output html file. Also target this file to be run-as-is.
Please Try Again.
"""

                with open(f"{self.base_dir}/{path}", "w") as f:
                    f.write(content)

                diff = difflib.unified_diff(old_content.splitlines(), content.splitlines(), fromfile=f"before/{path}", tofile=f"after/{path}")
                diff = "\n".join(diff)
                return f"File {path} updated successfully.Summary of Changes:\n\n{diff}"

        except Exception as e:
            return f"Failed to write file: {e}"


class ListFilesTool(BaseTool):
    name: str = "List Existing Files"
    id: str = "list_files"
    description: str = "List the files in current bucket."
    args_schema: Type[BaseModel] = ListFilesToolSchema
    base_dir: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, **kwargs) -> str:
        try:
            files = os.listdir(self.base_dir)
            files = [f for f in files if not f.startswith('.') and os.path.isfile(f"{self.base_dir}/{f}")]
            if not files:
                return "# -- Theres nothing to be listed -- #"
            return "\n".join(files)
        except Exception as e:
            return f"Failed to list files: {e}"


class SearchSoundTool(BaseTool):
    name: str = "search_sound"
    id: str = "search_sound"
    description: str = "Search for sounds using the FreeSound API."
    args_schema: Type[BaseModel] = SearchSoundToolSchema

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, **kwargs) -> List[Dict[str, Any]]:
        query = kwargs['query']
        min_duration = kwargs['min_duration']
        max_duration = kwargs['max_duration']
        max_results = kwargs.get('max_results', 8)

        client = FreesoundClient()
        client.set_token(os.environ.get('FREESOUND_CLIENT_API_KEY'), 'token')
        results = client.text_search(
            query=query, filter=f"duration:[{min_duration} TO {max_duration}]"
        )

        fetched_results = []
        for idx, sound in enumerate(results):
            if idx >= max_results:
                break

            sound_id = sound.id
            sound_user = sound.username
            sound_url = f"https://freesound.org/people/{sound_user}/sounds/{sound_id}/"

            page = requests.get(sound_url)
            soup = BeautifulSoup(page.content, 'html.parser')
            sound_description = soup.find(id="soundDescriptionSection")
            sound_description = re.sub(r'<.*?>', '', str(sound_description))

            fetched_results.append(
                {
                    "sound": sound,
                    "name": sound.name,
                    "description": sound_description,
                }
            )

        return fetched_results


class SaveSoundTool(BaseTool):
    name: str = "save_sound"
    id: str = "save_sound"
    description: str = "Save a sound file from FreeSound using the sound ID."
    args_schema: Type[BaseModel] = SaveSoundToolSchema

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, **kwargs) -> str:
        sound_id = kwargs['sound_id']
        file_name = kwargs['file_name']

        client = FreesoundClient()
        client.set_token(os.environ['FREESOUND_CLIENT_API_KEY'], 'token')

        current_directory = os.getcwd()
        try:
            chosen_sound = client.get_sound(sound_id)
            chosen_sound.retrieve_preview(current_directory, file_name)
            return f"Sound with id: {sound_id}, with name: {file_name}."
        except Exception as e:
            return f"Failed to save sound: {e}"

class ReadHtmlExamplesTool(BaseTool):
    name: str = "Read HTML examples"
    id: str = "read_examples_html"
    description: str = "Read all example html games."
    args_schema: Type[BaseModel] = ReadHtmlExamplesToolSchema
    base_dir: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, **kwargs) -> Dict[str, Dict[str, str]]:
        examples_dir = os.path.normpath(__file__ + '/../refs/html_game_examples')
        examples = {}

        try:
            for filename in os.listdir(examples_dir):
                file_path = os.path.join(examples_dir, filename)
                if filename.endswith('.html'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        examples[filename] = f.read()

            return examples
        except Exception as e:
            return f"Failed to read examples: {e}"

class QueryMechanicsTool(BaseTool):
    name: str = "Query Mechanics Tool"
    id: str = "query_mechanics"
    description: str = (
        "Searches a JSON file of game mechanics (with precomputed embeddings) for the closest "
        "semantic match to the query. Returns matching mechanics that fall within a given threshold."
    )
    args_schema: Type[BaseModel] = QueryMechanicsToolSchema

    _embeddings_file: str = PrivateAttr()
    _initial_top_k: int = PrivateAttr()
    _threshold: float = PrivateAttr()
    _mechanics: List[Any] = PrivateAttr()
    _embeddings: np.ndarray = PrivateAttr()
    _dimension: int = PrivateAttr()
    _index: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(self, initial_top_k: int = 15, threshold: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self._embeddings_file = os.path.normpath(__file__ + "/../refs/mechanics_db/mechanics_with_embeddings.json")
        self._initial_top_k = initial_top_k
        self._threshold = threshold

        # Load the mechanics JSON with embeddings
        try:
            with open(self._embeddings_file, 'r') as infile:
                self._mechanics = json.load(infile)
        except Exception as e:
            raise ValueError(f"Failed to load mechanics from {self._embeddings_file}: {e}")

        # Build a FAISS index from the precomputed embeddings
        try:
            self._embeddings = np.array([m['embedding'] for m in self._mechanics]).astype('float32')
            self._dimension = self._embeddings.shape[1]
            self._index = faiss.IndexFlatL2(self._dimension)
            self._index.add(self._embeddings)
        except Exception as e:
            raise ValueError(f"Failed to build FAISS index: {e}")

        # Initialize the SentenceTransformer for query encoding
        try:
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            raise ValueError(f"Failed to initialize SentenceTransformer: {e}")

    def _run(self, **kwargs) -> str:
        query = kwargs.get("query", "")
        if not query:
            return "No query provided."

        query_embedding = self._model.encode(query).astype('float32')
        query_embedding = np.expand_dims(query_embedding, axis=0)

        distances, indices = self._index.search(query_embedding, self._initial_top_k)

        relevant_results = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist < self._threshold:
                normalized_similarity = max(0, (self._threshold - dist) / self._threshold)
                mechanic = self._mechanics[idx]
                mechanic['similarity_score'] = round(normalized_similarity, 4)
                relevant_results.append(mechanic)

        if not relevant_results:
            return "No relevant game mechanics found for your query."

        response_lines = []
        for i, result in enumerate(relevant_results, 1):
            response_lines.append(f"Result {i}:")
            response_lines.append(f"Name: {result.get('Name', 'N/A')}")
            response_lines.append(f"Description: {result.get('Description', 'N/A')}")
            response_lines.append(f"Implementation Details: {result.get('Implementation Details', 'N/A')}")
            response_lines.append(f"Pseudocode: {result.get('Pseudocode (Phaser.js)', 'N/A')}")
            response_lines.append(f"Similarity Score: {result.get('similarity_score', 'N/A')}\n")
        return "\n".join(response_lines)

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)

class GoogleSearchTool(BaseTool):
    name: str = "Google Search"
    id: str = "google_search"
    description: str = "Search Google for recent results using the provided query."
    args_schema: Type[BaseModel] = GoogleSearchToolSchema

    search: GoogleSerperAPIWrapper = Field(default=None)  # Define search as a Pydantic field

    # Add model configuration for Pydantic
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'search', GoogleSerperAPIWrapper())  # Bypass Pydantic validation

    # Define the _run method to execute the search
    def _run(self, **kwargs) -> str:
        try:
            query = kwargs['query']
            results = self.search.run(query)
            return results
        except Exception as e:
            return f"Failed to perform Google search: {e}"

class ReadScaffoldTool(BaseTool):
    name: str = "Read Scaffold Tool"
    id: str = "read_scaffold"
    description: str = (
        "Retrieve the base scaffolding files for a platformer project. "
        "Use mode 'list' to view available files, or 'read' to read a specific file (or all files)."
    )
    args_schema: Type[BaseModel] = ReadScaffoldToolSchema
    base_dir: str

    _scaffold_path: str = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, **kwargs) -> str:
        try:
            # Determine the scaffold directory relative to this file.
            scaffold_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../refs/scaffold_platformer"))
            
            # Parse incoming arguments.
            args = self.args_schema.parse_obj(kwargs)
            mode = args.mode.lower()
            
            if mode == "list":
                # List all available scaffold files.
                files = os.listdir(scaffold_dir)
                if not files:
                    return "No scaffold files available."
                return "Available scaffold files:\n" + "\n".join(files)
            
            elif mode == "read":
                # If filename is not provided, default to "all".
                requested_file = args.filename if args.filename else "all"
                if requested_file.lower() == "all":
                    files = os.listdir(scaffold_dir)
                    if not files:
                        return "No scaffold files found."
                    contents = []
                    for filename in files:
                        file_path = os.path.join(scaffold_dir, filename)
                        with open(file_path, "r", encoding="utf-8") as f:
                            contents.append(f"----- {filename} -----\n{f.read()}\n")
                    return "\n".join(contents)
                else:
                    # Return the contents of the specified file.
                    file_path = os.path.join(scaffold_dir, requested_file)
                    if not os.path.exists(file_path):
                        return f"File '{requested_file}' does not exist in the scaffold folder."
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    return content
            else:
                return "Invalid mode specified. Please use 'list' or 'read'."
        except Exception as e:
            return f"Failed to read scaffold: {e}"

class SearchAndSaveSoundTool(BaseTool):
    """
    Searches Freesound for the first sound matching the query and minimum duration.
    Scrapes a short description from its webpage, saves the sound locally, and returns info as JSON.
    """
    name: str = "search_and_save_sound"
    id: str = "search_and_save_sound"
    description: str = "A formatted Freesound API query by using '+' for mandatory terms and '-' for prohibited ones (default is mandatory). For example, query=term_a -term_b excludes sounds with 'term_b'. Terms are separated by spaces. Generate a query based on your sound description. Avoid using the word sound in the query."
    args_schema: Type[BaseModel] = SearchAndSaveSoundToolSchema

    def _run(self, **kwargs) -> Any:
        import os
        import json
        import re
        import requests
        from bs4 import BeautifulSoup
        from freesound import FreesoundClient

        description = kwargs["description"]
        output_path = kwargs["output_path"]
        min_duration = kwargs.get("min_duration", 5)
        max_results = kwargs.get("max_results", 8)

        if not description or not output_path:
            return "Missing required fields: 'description' and 'output_path'."

        token = os.environ.get("FREESOUND_CLIENT_API_KEY")
        if not token:
            return "FREESOUND_CLIENT_API_KEY environment variable is not set."

        # Initialize the client
        client = FreesoundClient()
        client.set_token(token, "token")

        # Perform the search
        try:
            filter_str = f"duration:[{min_duration} TO *]"
            pager = client.text_search(query=description, filter=filter_str)
        except Exception as e:
            return f"Freesound search failed: {e}"

        # Collect the first result
        results = []
        for idx, sound in enumerate(pager):
            if idx >= max_results:
                break
            results.append(sound)

        if not results:
            return "No results found."

        # We'll just pick the first result to save
        chosen_sound = results[0]
        sound_id = chosen_sound.id
        sound_user = chosen_sound.username
        url = f"https://freesound.org/people/{sound_user}/sounds/{sound_id}/"

        # Scrape a short description
        try:
            page = requests.get(url)
            soup = BeautifulSoup(page.content, "html.parser")
            desc_section = soup.find(id="soundDescriptionSection")
            raw_desc = re.sub(r"<.*?>", "", str(desc_section)) if desc_section else ""
        except Exception:
            raw_desc = "N/A"

        # Save the preview locally
        try:
            directory = os.path.dirname(output_path)
            filename = os.path.basename(output_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            chosen_sound.retrieve_preview(directory, filename)
        except Exception as e:
            return f"Failed to save sound (ID={sound_id}): {e}"

        # Build the response
        response_data = {
            "chosen_sound_id": sound_id,
            "name": chosen_sound.name,
            "description": raw_desc.strip(),
            "saved_path": output_path
        }
        return json.dumps(response_data, indent=2)

    async def _arun(self, **kwargs) -> Any:
        return self._run(**kwargs)

dalle_tool = DallETool(model="dall-e-3",
                       size="1024x1024",
                       quality="standard",
                       n=1)

class GenerateAndDownloadImageTool(BaseTool):
    """
    A single tool that generates an image using OpenAI's DALL·E API and downloads it locally.
    """
    name: str = "generate_and_download_image"
    id: str = "generate_and_download_image"
    description: str = (
        "Generate an image from a prompt via DALL·E, then download the resulting image to file."
    )
    args_schema: Type[BaseModel] = GenerateAndDownloadImageSchema

    def _run(self, **kwargs) -> Any:
        prompt = kwargs["prompt"]
        file_name = kwargs["file_name"]
        n = 1
        size = kwargs.get("size", "1024x1024")
        response_format = kwargs.get("response_format", "url")
        # model = kwargs.get("model", "image-alpha-001") # If you want a specific model param

        # Make sure your OPENAI_API_KEY is set
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            return "OPENAI_API_KEY is not set in the environment."

        try:
            # Configure OpenAI
            openai.api_key = openai_api_key
            client = OpenAI(api_key=openai_api_key)

            response = client.images.generate(
                prompt=prompt,
                n=n,
                size=size,
                response_format=response_format,
                model="dall-e-3", 
            )

            # We'll just take the first generated image
            # If response_format="url", we get a URL for the image.
            # If "b64_json", we get a base64-encoded string.
            response_dict = response.model_dump(mode="python")
            if not response_dict or "data" not in response_dict or len(response_dict["data"]) == 0:
                return "No image data returned from DALL·E."
            image_url = response_dict["data"][0]["url"]

            # Depending on the response format, extract the image data
            if response_format == "url":
                image_url = response_dict["data"][0]["url"]
                # Download the image from the URL
                r = requests.get(image_url)
                r.raise_for_status()  # Raise an error if the HTTP request failed
                with open(file_name, "wb") as f:
                    f.write(r.content)
                return json.dumps({
                    "message": f"Image generated and saved as {file_name}",
                    "url": image_url
                }, indent=2)
            else:
                # If response_format is "b64_json", decode the base64 data and write it to file
                b64_data = response_dict["data"][0]["b64_json"]
                image_bytes = b64_data.encode("utf-8")  # Convert string to bytes
                import base64
                decoded = base64.decodebytes(image_bytes)
                with open(file_name, "wb") as f:
                    f.write(decoded)
                return json.dumps({
                    "message": f"Image generated (base64) and saved as {file_name}"
                }, indent=2)

        except Exception as e:
            return f"Image generation or download failed: {e}"

    async def _arun(self, **kwargs) -> Any:
        return self._run(**kwargs)
    
def get_all_tools():
    # base_dir = TemporaryDirectory(delete=False).name
    base_dir = "."

    # print(f"Temp directory created at: {base_dir}")

    def no_cache(args, result):
        return False

    tools = {}
    toolklasses = [
        ReadFileTool, BatchReadFilesTool, WriteFileTool, ListFilesTool,
        SearchAndSaveSoundTool, GenerateAndDownloadImageTool, ReadHtmlExamplesTool, QueryMechanicsTool, GoogleSearchTool, ReadScaffoldTool
    ]
    for toolkls in toolklasses:
        tool = toolkls(base_dir=base_dir)
        tool.cache_function = no_cache
        tools[tool.id] = tool

    return tools