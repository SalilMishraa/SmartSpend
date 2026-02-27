from fastapi import APIRouter, HTTPException
from ...schemas.analysis import AnalyzeRequest, AnalyzeResponse

from services.parser import process_transactions
from services.metrics import compute_metrics
from services.anomalies import detect_anomalies
from services.ai import generate_ai_suggestions

import pandas as pd
import io

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):

    try:
        # Convert CSV string to DataFrame
        df = pd.read_csv(io.StringIO(request.raw_data))

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

        # Generate AI suggestions
        ai_suggestions = generate_ai_suggestions(
            metrics,
            request.spending_limit
        )

        # Remove internal-only data
        metrics.pop("daily_spend_df", None)

        return {
            "metrics": metrics,
            "ai_suggestions": ai_suggestions,
            "dropped_rows": dropped_rows
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")