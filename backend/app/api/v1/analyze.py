from fastapi import APIRouter, HTTPException, Request
from ...schemas.analysis import AnalyzeRequest, AnalyzeResponse

from services.parser import process_transactions
from services.metrics import compute_metrics
from services.anomalies import detect_anomalies
from services.ai import generate_ai_suggestions

import pandas as pd
import io

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request_data: AnalyzeRequest, request: Request):

    try:
        # Convert CSV string to DataFrame
        df = pd.read_csv(io.StringIO(request_data.raw_data))

        # Parse transactions
        expenses_df, dropped_rows = process_transactions(df)

        # Compute metrics
        metrics = compute_metrics(expenses_df)

        # Detect anomalies
        metrics["anomalies"] = detect_anomalies(
            metrics["daily_spend_df"],
            metrics["category_spending"],
            expenses_df
        )

        # Remove internal-only dataframe before serialization
        metrics.pop("daily_spend_df", None)

        # Get Groq client from app state
        groq_client = request.app.state.groq_client

        # Generate AI suggestions
        ai_suggestions = generate_ai_suggestions(
            metrics,
            request_data.spending_limit,
            groq_client
        )

        # Convert datetime objects to strings
        for item in metrics.get("daily_spending", []):
            if hasattr(item["date"], "strftime"):
                item["date"] = item["date"].strftime("%Y-%m-%d")

        for item in metrics.get("top_3_days", []):
            if hasattr(item["date"], "strftime"):
                item["date"] = item["date"].strftime("%Y-%m-%d")

        for item in metrics.get("anomalies", []):
            if hasattr(item["date"], "strftime"):
                item["date"] = item["date"].strftime("%Y-%m-%d")

        return {
            "metrics": metrics,
            "ai_suggestions": ai_suggestions,
            "dropped_rows": dropped_rows
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")