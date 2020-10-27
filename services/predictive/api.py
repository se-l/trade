from fastapi import APIRouter

router = APIRouter()


@router.get("/predictive/", tags=["predictive"])
async def get_prediction():
    return [{"username": "Foo"}, {"username": "Bar"}]
