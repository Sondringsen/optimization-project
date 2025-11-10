import os
from openevolve import evolve_function
from openevolve.config import Config, LLMModelConfig
import asyncio
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY", "")
config = Config()
config.llm.models = [LLMModelConfig(name='gpt-4', api_key=openai_api_key)]

# Evolve Python functions directly
def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j] 
    return arr

result = evolve_function(
    bubble_sort,
    test_cases=[([3,1,2], [1,2,3]), ([5,2,8], [2,5,8])],
    config=config,
    iterations=5
)
print(f"Evolved sorting algorithm: {result.best_code}")

