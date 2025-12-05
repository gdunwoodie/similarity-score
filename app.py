from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import shutil

# IMPORT YOUR EXTRACTOR
# Make sure feature_extractor.py is in the same folder as app.py
from feature_extractor import extract_features_from_path 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    # 1. Create a temp file with the correct extension (e.g., .wav)
    # Windows requires us to close the file before librosa can open it.
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        # 2. Run your extractor logic
        features = extract_features_from_path(tmp_path)
        
        # 3. WRAP THE RESPONSE
        # This solves the "undefined" error
        return {"features": features} 

    except Exception as e:
        print(f"Server Error: {e}") # Print error to terminal
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 4. Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=5000)

@app.get("/")
def read_root():
    return {"status": "Backend is running!"}
