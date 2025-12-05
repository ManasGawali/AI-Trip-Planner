# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from agent.agentic_workflow import GraphBuilder
# from utils.save_to_document import save_document
# from starlette.responses import JSONResponse
# import os
# import datetime
# from dotenv import load_dotenv
# from pydantic import BaseModel
# load_dotenv()

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # set specific origins in prod
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# class QueryRequest(BaseModel):
#     question: str

# @app.post("/query")
# async def query_travel_agent(query:QueryRequest):
#     try:
#         print(query)
#         graph = GraphBuilder(model_provider="groq")
#         react_app=graph()
#         #react_app = graph.build_graph()

#         png_graph = react_app.get_graph().draw_mermaid_png()
#         with open("my_graph.png", "wb") as f:
#             f.write(png_graph)

#         print(f"Graph saved as 'my_graph.png' in {os.getcwd()}")
#         # Assuming request is a pydantic object like: {"question": "your text"}
#         messages={"messages": [query.question]}
#         output = react_app.invoke(messages)

#         # If result is dict with messages:
#         if isinstance(output, dict) and "messages" in output:
#             final_output = output["messages"][-1].content  # Last AI response
#         else:
#             final_output = str(output)
        
#         return {"answer": final_output}
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from agent.agentic_workflow import GraphBuilder
from utils.save_to_document import save_document
from starlette.responses import JSONResponse
import os
import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
from pathlib import Path
import shutil

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set specific origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_travel_agent(query: QueryRequest):
    """
    Endpoint for text-based travel queries
    """
    try:
        print(f"Received query: {query.question}")
        graph = GraphBuilder(model_provider="groq")
        react_app = graph()

        # Save graph visualization
        png_graph = react_app.get_graph().draw_mermaid_png()
        with open("my_graph.png", "wb") as f:
            f.write(png_graph)

        print(f"Graph saved as 'my_graph.png' in {os.getcwd()}")
        
        # Process the query
        messages = {"messages": [query.question]}
        output = react_app.invoke(messages)

        # Extract final response
        if isinstance(output, dict) and "messages" in output:
            final_output = output["messages"][-1].content
        else:
            final_output = str(output)
        
        return {"answer": final_output}
    
    except Exception as e:
        print(f"Error in query endpoint: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/recognize-landmark")
async def recognize_landmark(file: UploadFile = File(...)):
    """
    Endpoint for landmark recognition from uploaded image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Create temp directory if it doesn't exist
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded file temporarily
        temp_file_path = temp_dir / f"temp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Image saved temporarily at: {temp_file_path}")
        
        # Import and use landmark recognition tool
        try:
            from tools.landmark_recognition_tool import landmark_recognition_tool
            
            # Recognize the landmark
            result = landmark_recognition_tool(str(temp_file_path))
            
            # Clean up temp file
            if temp_file_path.exists():
                os.remove(temp_file_path)
            
            if result['success']:
                # Generate trip plan query
                travel_query = (
                    f"Plan a trip to {result['landmark_name']} in {result['city']}, {result['country']}. "
                    f"Include best time to visit, things to do, places to see nearby, estimated costs, "
                    f"and a 3-day itinerary."
                )
                
                # Get trip plan from agent
                graph = GraphBuilder(model_provider="groq")
                react_app = graph()
                messages = {"messages": [travel_query]}
                output = react_app.invoke(messages)
                
                if isinstance(output, dict) and "messages" in output:
                    trip_plan = output["messages"][-1].content
                else:
                    trip_plan = str(output)
                
                # Combine landmark info with trip plan
                result['trip_plan'] = trip_plan
                
                return result
            else:
                return result
        
        except ImportError:
            # Clean up temp file
            if temp_file_path.exists():
                os.remove(temp_file_path)
            
            raise HTTPException(
                status_code=500, 
                detail="Landmark recognition model not available. Please train the model first."
            )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in landmark recognition endpoint: {str(e)}")
        
        # Clean up temp file in case of error
        if 'temp_file_path' in locals() and temp_file_path.exists():
            os.remove(temp_file_path)
        
        return JSONResponse(
            status_code=500, 
            content={"error": f"Landmark recognition failed: {str(e)}"}
        )


@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Travel Planner API",
        "version": "2.0",
        "endpoints": {
            "/query": "POST - Text-based travel queries",
            "/recognize-landmark": "POST - Image-based landmark recognition",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)