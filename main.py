from fastapi import FastAPI
import uvicorn
from fastapi.responses import JSONResponse


app = FastAPI()




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
