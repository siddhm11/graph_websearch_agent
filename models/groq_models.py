import requests
import json
import os
import time
from utils.helper_functions import load_config
from langchain_core.messages.human import HumanMessage


class GroqJSONModel:
    def __init__(self, temperature=0, model=None):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        load_config(config_path)
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.headers = {
            'Content-Type': 'application/json', 
            'Authorization': f'Bearer {self.api_key}'
            }
        self.model_endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.temperature = temperature
        self.model = model

    def invoke(self, messages):

        system = messages[0]["content"]
        user = messages[1]["content"]

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": f"system:{system}\n\n user:{user}"
                }
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"}
        }
        
        try:
            request_response = requests.post(
                self.model_endpoint, 
                headers=self.headers, 
                data=json.dumps(payload)
            )
            
            print("REQUEST RESPONSE", request_response.status_code)
            # print("REQUEST RESPONSE HEADERS", request_response.headers)
            # print("REQUEST RESPONSE TEXT", request_response.text)
            
            request_response_json = request_response.json()
            # print("REQUEST RESPONSE JSON", request_response_json)
            
            if 'choices' not in request_response_json or len(request_response_json['choices']) == 0:
                raise ValueError("No choices in response")

            response_content = request_response_json['choices'][0]['message']['content']
            # print("RESPONSE CONTENT", response_content)
            
            response = json.loads(response_content)
            response = json.dumps(response)

            response_formatted = HumanMessage(content=response)

            return response_formatted
        except (requests.RequestException, ValueError, KeyError) as e:
            error_message = f"Error in invoking model! {str(e)}"
            print("ERROR", error_message)
            response = {"error": error_message}
            response_formatted = HumanMessage(content=json.dumps(response))
            return response_formatted

class GroqModel:
    def __init__(self, temperature=0, model=None):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        load_config(config_path)
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.headers = {
            'Content-Type': 'application/json', 
            'Authorization': f'Bearer {self.api_key}'
            }
        self.model_endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.temperature = temperature
        self.model = model

    def invoke(self, messages):
        time.sleep(0.5)   # wait for half a second

        system = messages[0]["content"]
        user = messages[1]["content"]

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": f"system:{system}\n\n user:{user}"
                }
            ],
            "temperature": self.temperature,
        }

        try:
            request_response = requests.post(
                self.model_endpoint, 
                headers=self.headers, 
                data=json.dumps(payload)
                )
            
            
            print("REQUEST RESPONSE", request_response.status_code)
            request_response_json = request_response.json()['choices'][0]['message']['content']
            resp = request_response.json()
            print("🔍 [DEBUG] Groq response status:", request_response.status_code)
            print("🔍 [DEBUG] Groq response body:", resp)

            # 1) HTTP status check
            if request_response.status_code != 200:
                raise RuntimeError(f"LLM API returned HTTP {request_response.status_code}: {resp}")

            # 2) Ensure choices exist
            if 'choices' not in resp or not resp['choices']:
                raise RuntimeError(f"No choices in response: {resp}")

            # 3) Safely extract the assistant’s content
            content = resp['choices'][0]['message']['content']

            response_formatted = HumanMessage(content=content)

            response = str(request_response_json)            
            response_formatted = HumanMessage(content=response)

            return response_formatted
        except requests.RequestException as e:
            response = {"error": f"Error in invoking model! {str(e)}"}
            response_formatted = HumanMessage(content=response)
            return response_formatted
