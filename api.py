from llama_cpp import Llama
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import json

api = FastAPI()

origins = [
    os.getenv("ORIGIN"),
]

api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
  model_path="../llamAPI/models/" + os.getenv("LLAMA_MODEL"),  # Download the model file first
  n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
  n_gpu_layers=24         # The number of layers to offload to GPU, if you have GPU acceleration available
)

inUse = False

@api.get("/status")
async def get_status( request: Request ):
  return { "model_loaded": True, "avaliable": !inUse }  

@api.post("/generate")
async def generate_text( request: Request):
  if inUse:
      return { "error": "Node not yet available" }

  inUse = True
  data = await request.json()
  print( json.dumps( data["data"], indent=1))
  prompt = data.get("data", "")["package"]

  if not prompt:
    return {"error": "package is required"}
  

  print("Generating for prompt: ", prompt)

  # Simple inference example
  output = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": """Narrative Statements are a narrative style used to communicate accomplishments and results in the United States Air Force. They should be efficient and increase clarity of an Airman's performance.
                In the United States Air Force, Narrative Statements should be a standalone sentence with action and at least one of impact or results/outcome and written in plain language without uncommon acronyms and abbreviations.
                The first word of a narrative statement should be a strong action verb.
                The performance statement should be one sentence and written in past tense. It should also include transition words like "by" and "which".
                Personal pronouns (I, me, my, we, us, our, etc.) should not be used.
                Rewrite the USER prompt to follow these conventions. 
                Generate Three seporate and unique ways to rewrite what you were given in JSON format labled "V1", "V2", and "V3", and how it has improved in "V1_Reason", "V2_Reason", and "V3_Reason".
                At the very end generate impartial feedback on how to improve the statement in JSON labled "Feedback" """,
        },
        {"role": "user", "content": prompt},
    ],
    response_format={
      "type": "json_object",
      "schema": {
            "type": "object",
            "properties": {
              "V1": {
                  "type": "string"
              },
              "V2": {
                  "type": "string"
              },
              "V3": {
                  "type": "string"
              },
              "V1_Reason": {
                  "type": "string"
              },
              "V2_Reason": {
                  "type": "string"
              },
              "V3_Reason": {
                  "type": "string"
              },
              "Feedback": {
                  "type": "string"
              }
            },
            "required": ["V1", "V2", "V3", "V1_Reason", "V2_Reason", "V3_Reason", "Feedback"]
      }
    },
    temperature=0.7
    # max_tokens=512,  # Generate up to 512 tokens
    # stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
    # echo=True        # Whether to echo the prompt
  )
  inUse = False

  return {"Feedback": output["choices"][0]["message"]["content"]}
