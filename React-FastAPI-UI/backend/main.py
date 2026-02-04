from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import ChatRequest, ChatResponse
from vertex_agent import query_agent_engine

app = FastAPI(title="Vertex AI Agent Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        response_text = query_agent_engine(
            user_message=request.user_message,
            agent_id=request.agent_id,
            project_id=request.project_id,
            location=request.location,
        )
        return ChatResponse(response_text=response_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__=="__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )