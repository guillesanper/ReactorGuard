"""SHAP-based explanation endpoint."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class ExplanationRequest(BaseModel):
    prediction_id: str


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    sensor_value: float


class ExplanationResponse(BaseModel):
    prediction_id: str
    top_features: list[FeatureContribution]
    base_value: float


@router.post("/", response_model=ExplanationResponse)
async def explain(request: ExplanationRequest) -> ExplanationResponse:
    """Return SHAP feature contributions for a previous prediction."""
    # TODO: retrieve prediction context, compute SHAP values
    raise HTTPException(status_code=501, detail="Explainability not yet implemented")
