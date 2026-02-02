def detect_anomalies(daily_spend_df, category_spending, expenses_df):
    """
    Detect spending anomalies:
    1. Daily total > 2× average daily spend
    2. Category daily spend > 1.5× category's average daily spend
    """

    anomalies = []

    # --- Daily total anomalies ---
    if daily_spend_df.empty:
        return anomalies

    avg_daily = daily_spend_df['amount'].mean()

    for _, row in daily_spend_df.iterrows():
        if row['amount'] > 2 * avg_daily:
            anomalies.append({
                "type": "daily",
                "date": row['date'],
                "category": "Overall",
                "amount": row['amount'],
                "reason": "Daily spend > 2× average daily spend"
            })

    # --- Category daily anomalies ---
    if expenses_df.empty:
        return anomalies

    # Daily spend per category
    cat_day = (
        expenses_df
        .groupby(['category', 'date'])['amount']
        .sum()
        .reset_index()
    )

    # Average daily spend per category
    cat_avg = (
        cat_day
        .groupby('category')['amount']
        .mean()
        .to_dict()
    )

    for _, row in cat_day.iterrows():
        category = row['category']
        amount = row['amount']
        avg_cat = cat_avg.get(category, 0)

        if avg_cat > 0 and amount > 1.5 * avg_cat:
            anomalies.append({
                "type": "category",
                "date": row['date'],
                "category": category,
                "amount": amount,
                "reason": "Category daily spend > 1.5× its average"
            })

    return anomalies
