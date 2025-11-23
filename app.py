from fastapi import FastAPI,Path
from fastapi.responses import JSONResponse,StreamingResponse
from pydantic import BaseModel,Field,field_validator,field_serializer
from typing import Annotated,List,Dict,Literal
from langchain_core.messages import HumanMessage
from langchain.load.dump import dumps
from project_4 import agent,retrive_all_threads
import uuid

class prompt_request(BaseModel):
    thread_id:str = Field(...,description="the thread_id of the requied or current chat session")
    input:str = Field(...,description="the prompt or input to the model or openAi")
    mode:Literal['invoke','stream'] = Field(...,description="the mode of the output")

class get_state_scheme(BaseModel):
    thread_id:str

app = FastAPI()

@app.get("/")
def fun1():
    return {"message":"this a experimental project"}

@app.post("/prompt")
def prompt(prompt: prompt_request):
    try:
        CONFIG = {"configurable": {'thread_id': prompt.thread_id}}
        if prompt.mode == "invoke":
            res = JSONResponse(
                status_code=200,
                content={"response": dumps(agent.invoke({"messages": [HumanMessage(content=prompt.input)]}, config=CONFIG))}
            )
        else:
            def stream_gen():
                for chunk in agent.stream(
                    {"messages": [HumanMessage(content=prompt.input)]},
                    config=CONFIG,
                    stream_mode="messages"
                ):
                    yield dumps(chunk) + "\n"
            res = StreamingResponse(stream_gen(), media_type="application/json")
        return res
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.get("/get_all_threads")
def threads():
    try:
        return JSONResponse(status_code=200,content={"threads":retrive_all_threads()} )
    except Exception as e:
        return JSONResponse(status_code=500,content={"error":str(e)})
    
@app.get("/uuid")
def gen_rand_thread_id():
    return JSONResponse(status_code=200,content={"uuid":str(uuid.uuid4())})

@app.post("/get_state")
def get_state(input:get_state_scheme):
    print(input.thread_id)
    print(agent.get_state(config={'configurable':{"thread_id":input.thread_id}}))
    return JSONResponse(status_code = 200,content= {"state":dumps(agent.get_state(config={'configurable':{"thread_id":input.thread_id}}).values.get("messages",[]))})