# src/mcp_cli/llm/providers/gemini_client.py
import os
import logging
import json
from typing import Any, Dict, List
from dotenv import load_dotenv

# Use 'from google import generativeai' import style
from google import genai
# Import specific types needed
from google.genai import types
# from google.generativeai.types import FunctionDeclaration, Tool, GenerationConfig, Part # Keep commented or remove if unused
from mcp_cli.llm.providers.base import BaseLLMClient

load_dotenv()

# Removed DEFAULT_SAFETY_SETTINGS

class GeminiLLMClient(BaseLLMClient):
    # Removed safety_settings parameter from __init__
    def __init__(self, model="gemini-1.5-flash-latest", api_key=None):
        self.model_name = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        # Removed self.safety_settings assignment

        if not self.api_key:
            raise ValueError("The GEMINI_API_KEY environment variable is not set.")

        try:
            # Initialize the client using the API key
            self.client = genai.Client(api_key=self.api_key)
            # Removed direct model instantiation here
            logging.info(f"Gemini client initialized for model: {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini client: {e}")
            raise ValueError(f"Failed to initialize Gemini client: {e}")

    def _convert_messages_to_gemini_format(self, messages: List[Dict]) -> List[types.ContentDict]:
        """
        Converts the standard message format (OpenAI-like) to Gemini's Content format.
        Handles user, assistant (model), system, and tool messages.
        """
        gemini_contents: List[types.ContentDict] = []
        system_instruction = None

        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content") # Text content
            tool_calls = msg.get("tool_calls") # Model's request to call tools
            tool_call_id = msg.get("tool_call_id") # ID for tool response

            gemini_role = None
            parts = []

            if role == "system":
                system_instruction = content
                logging.debug("Storing system instruction for Gemini.")
                continue # Skip adding as a separate message for now

            elif role == "user":
                gemini_role = "user"
                if content:
                    parts.append(types.Part(text=content))

            elif role == "assistant":
                gemini_role = "model"
                if content:
                    parts.append(types.Part(text=content))
                if tool_calls:
                    for tc in tool_calls:
                        # Ensure arguments are parsed correctly before creating FunctionCall
                        try:
                            args_dict = json.loads(tc.get("function", {}).get("arguments", "{}"))
                        except json.JSONDecodeError:
                            logging.error(f"Failed to parse tool call arguments: {tc.get('function', {}).get('arguments')}")
                            args_dict = {} # Fallback

                        function_call_part = types.FunctionCall(
                            name=tc.get("function", {}).get("name"),
                            args=args_dict
                        )
                        parts.append(types.Part(function_call=function_call_part))

            elif role == "tool":
                 gemini_role = "function" # Use 'function' role for tool responses
                 if tool_call_id and content:
                     tool_name = "unknown_tool" # Fallback
                     if i > 0 and messages[i-1].get("role") == "assistant":
                         prev_tool_calls = messages[i-1].get("tool_calls", [])
                         # Attempt to match tool_call_id to the generated ID from previous turn
                         # This is still fragile as Gemini doesn't provide IDs
                         matched_call = next((ptc for ptc in prev_tool_calls if ptc.get("id") == tool_call_id), None)
                         if matched_call:
                             tool_name = matched_call.get("function", {}).get("name", tool_name)
                         else:
                             # Fallback: try getting name from content if JSON
                             try:
                                 tool_response_data = json.loads(content)
                                 tool_name = tool_response_data.get("name", tool_name)
                             except json.JSONDecodeError:
                                 logging.warning(f"Could not determine tool name for tool ID {tool_call_id}. Using fallback.")
                                 # Last resort: use name from first tool call in previous message
                                 if prev_tool_calls:
                                     tool_name = prev_tool_calls[0].get("function", {}).get("name", tool_name)

                     function_response_part = types.FunctionResponse(
                         name=tool_name,
                         response={"content": content}
                     )
                     parts.append(types.Part(function_response=function_response_part))
                 else:
                     logging.warning(f"Skipping tool message without tool_call_id or content: {msg}")
                     continue

            else:
                logging.warning(f"Unsupported role for Gemini conversion: {role}")
                continue

            if parts:
                gemini_contents.append(types.ContentDict(role=gemini_role, parts=parts))

        # Note: System instruction handling might need adjustment based on model/API version
        # if system_instruction: ...

        return gemini_contents

    # Modified to return List[types.Tool] or None as per example
    def _format_mcp_tools_for_gemini(self, mcp_tools: List[Dict]) -> List[types.Tool] | None:
        """Converts MCP tool schema list to a list of Gemini Tool objects."""
        if not mcp_tools:
            return None
        
        gemini_tools_list = []
        for mcp_tool in mcp_tools:
            if mcp_tool.get("type") == "function" and "function" in mcp_tool:
                func_data = mcp_tool["function"]
                # Filter schema as per example if needed, or use directly
                parameters_schema_dict = func_data.get("parameters", {"type": "object", "properties": {}})
                # Example filter:

                func_decl = types.FunctionDeclaration(
                    name=func_data.get("name"),
                    description=func_data.get("description", ""),
                    parameters=parameters_schema_dict, # Pass the raw (potentially filtered) dictionary
                )
                # Each MCP tool becomes a separate Tool object containing one declaration
                gemini_tools_list.append(types.Tool(function_declarations=[func_decl]))
            else:
                logging.warning(f"Skipping tool with unexpected format: {mcp_tool}")

        return gemini_tools_list if gemini_tools_list else None


    def create_completion(self, messages: List[Dict], tools: List = None) -> Dict[str, Any]:
        # Format tools for Gemini - returns List[Tool] or None
        gemini_tools_list = self._format_mcp_tools_for_gemini(tools)
        # Initialize GenerationConfig *with* the list of Tool objects if they exist.

        # Convert the full message history
        gemini_contents = self._convert_messages_to_gemini_format(messages)
        
        if not gemini_contents:
             logging.error("Could not convert messages to Gemini format.")
             return {"response": "Error: Could not process messages.", "tool_calls": []}

        try:
            logging.debug(f"Sending content to Gemini ({self.model_name})...")
            # Pass the GenerationConfig object containing tools list to the 'config' parameter
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=gemini_contents,
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=gemini_tools_list,
                ), # Pass the config object
            )

            # Handle potential blocks or errors in response
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason
                 block_message = response.prompt_feedback.block_reason_message or "No specific message."
                 error_message = f"Gemini prompt blocked. Reason: {block_reason}, Message: {block_message}"
                 logging.error(error_message)
                 return {"response": f"Error: {error_message}", "tool_calls": []}

            if not response.candidates:
                 finish_reason_str = getattr(response, 'prompt_feedback', 'Unknown finish reason')
                 error_message = f"Gemini response has no candidates. Finish reason: {finish_reason_str}"
                 logging.error(error_message)
                 return {"response": f"Error: {error_message}", "tool_calls": []}


            # --- Process Response: Extract Text and Tool Calls ---
            response_text = ""
            final_tool_calls = []
            candidate = response.candidates[0]

            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text is not None:
                        response_text += part.text
                    elif hasattr(part, 'function_call'):
                        func_call = part.function_call
                        try:
                            args_json = json.dumps(func_call.args) if func_call.args else "{}"
                        except TypeError as e:
                             logging.error(f"Could not serialize Gemini function call args to JSON: {func_call.args} - Error: {e}")
                             args_json = "{}"

                        call_id = f"call_{func_call.name}_{os.urandom(4).hex()}"
                        final_tool_calls.append({
                            "id": call_id,
                            "function": {
                                "name": func_call.name,
                                "arguments": args_json,
                            },
                        })
                    else:
                         logging.warning(f"Unhandled part type in Gemini response: {type(part)}")

            if not response_text and not final_tool_calls:
                 finish_reason = getattr(candidate, 'finish_reason', 'UNKNOWN')
                 if finish_reason != "STOP" and finish_reason != "TOOL_CALLS":
                     logging.warning(f"Gemini response is empty. Finish Reason: {finish_reason}")

            logging.debug(f"Received response text from Gemini: {response_text[:100]}...")
            if final_tool_calls:
                logging.debug(f"Received tool calls from Gemini: {final_tool_calls}")

            return {
                "response": response_text or None,
                "tool_calls": final_tool_calls
            }
        except Exception as e:
            logging.error(f"Gemini API Error: {str(e)}")
            error_detail = str(e)
            raise ValueError(f"Gemini API Error: {error_detail}")
