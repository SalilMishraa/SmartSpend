from fastapi import APIRouter

router = APIRouter()

@router.post("/analyze")
def analyze():
    return {
        "message": "Analyze endpoint wired correctly"
    }
