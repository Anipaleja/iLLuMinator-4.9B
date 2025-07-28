import uvicorn
from fastapi import FastAPI
from .model import create_illuminator_model

class iLLuMinatorAPI:
    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self.app = FastAPI(title="iLLuMinator AI API")
        
        @self.app.get("/")
        async def root():
            return {"message": "iLLuMinator AI API", "status": "running"}
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        uvicorn.run(self.app, host=host, port=port, **kwargs)

def main():
    api = iLLuMinatorAPI()
    api.run()

if __name__ == "__main__":
    main()
