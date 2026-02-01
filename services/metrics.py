import pandas as pd

def compute_metrics(expenses_df):
    """
    Compute all spending metrics from a cleaned expenses dataframe.

    Returns:
        dict containing spending metrics
    """
    metrics = {}

    # ---- Total spent ----
    metrics['total_spent'] = expenses_df['amount'].sum()

    # ---- Category-wise spending ----
    category_spending = (
        expenses_df
        .groupby('category')['amount']
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )
    metrics['category_spending'] = category_spending

    # ---- Daily spending ----
    daily_spending = (
        expenses_df
        .groupby('date')['amount']
        .sum()
        .sort_index()
    )

    metrics['daily_spend_df'] = daily_spending.reset_index()
    metrics['daily_spending'] = [
        {'date': d.strftime('%Y-%m-%d'), 'amount': a}
        for d, a in daily_spending.items()
    ]

    # ---- Weekly spending ----
    weekly_spending = (
        expenses_df
        .set_index('date')['amount']
        .resample('W-MON', label='left')
        .sum()
    )

    metrics['weekly_spending'] = (
        weekly_spending.to_dict() if not weekly_spending.empty else {}
    )

    # ---- Average daily spending ----
    metrics['avg_daily_spending'] = (
        daily_spending.mean() if not daily_spending.empty else 0
    )

    # ---- Top 3 spending days ----
    top_days = daily_spending.sort_values(ascending=False).head(3)
    metrics['top_3_days'] = [
        {'date': d.strftime('%Y-%m-%d'), 'amount': a}
        for d, a in top_days.items()
    ]

    return metrics
