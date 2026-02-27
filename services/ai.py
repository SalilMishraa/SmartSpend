from typing import Dict, Any, List


def generate_ai_suggestions(
    metrics: Dict[str, Any],
    limit: float,
    groq_client
) -> str:
    """Generate AI suggestions based on spending metrics."""

    if groq_client is None:
        return "AI suggestions are unavailable. Configure your Groq API key."

    category_spending = metrics.get("category_spending", {})
    total_spent = metrics.get("total_spent", 0)
    avg_daily = metrics.get("avg_daily_spending", 0)
    daily_spending = metrics.get("daily_spending", [])
    top_days = metrics.get("top_3_days", [])
    anomalies = metrics.get("anomalies", [])

    if not category_spending:
        return "AI suggestions are unavailable. Check your uploaded data."

    # Build category summary
    category_summary = "\n".join(
        f"- {category}: ₹{amount:,.2f}"
        for category, amount in category_spending.items()
    )

    # Budget calculations
    remaining = limit - total_spent
    budget_status = (
        f"₹{remaining:,.2f} remaining"
        if remaining > 0
        else f"₹{abs(remaining):,.2f} over budget"
    )

    days_tracked = len(daily_spending)

    # Original prompt style preserved
    prompt = (
        "You are a friendly financial advisor for a college student in India. "
        "Analyze the following spending data and provide 3-4 concise, actionable tips to help them meet their goal. "
        "Use markdown for formatting. (DO NOT USE TITLES OR HEADING, Just Bold text wherever a heading is required) "
        "Use a newline symbol after a bold heading \n\n"
        f"**Spending Goal:** ₹{limit:,.2f}\n"
        f"**Total Spent:** ₹{total_spent:,.2f}\n"
        f"**Budget Status:** {budget_status}\n"
        f"**Average Daily Spending:** ₹{avg_daily:,.2f}\n"
        f"**Days Tracked:** {days_tracked}\n"
    )

    if top_days:
        prompt += "\n**Top Spending Days:**\n"
        for i, day in enumerate(top_days, 1):
            prompt += f"{i}. {day['date']}: ₹{day['amount']:,.2f}\n"

    if anomalies:
        prompt += f"\n**Unusual Transactions Detected:** {len(anomalies)}\n"

    prompt += "\n**Spending by Category:**\n" + category_summary

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are SmartSpend, a budgeting assistant for college students in India. "
                        "Budget is a spending cap, not income. "
                        "All totals and amounts are pre-calculated—use them as-is. "
                        "Give practical advice with whole numbers (e.g., '3 meals' not '2.7 meals'). "
                        "Be concise and realistic. "
                        "Use markdown for formatting. "
                        "(DO NOT USE TITLES OR HEADING, Just Bold text wherever a heading is required). "
                        "Always add a newline after a bold heading."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Sorry, an API error occurred: {e}"


def get_chatbot_response(
    question: str,
    metrics: Dict[str, Any],
    chat_history: List[Dict[str, str]],
    groq_client
) -> str:
    """Generate chatbot response based on user question and financial context."""

    if groq_client is None:
        return "Chatbot is unavailable: Groq API key not configured."

    category_spending = metrics.get("category_spending", {})
    total_spent = metrics.get("total_spent", 0)
    avg_daily = metrics.get("avg_daily_spending", 0)
    daily_spending = metrics.get("daily_spending", [])

    category_summary = "\n".join(
        f"- {category}: ₹{amount:,.2f}"
        for category, amount in category_spending.items()
    )

    system_msg = {
        "role": "system",
        "content": (
            "You are SmartSpend, a budgeting assistant for college students in India. "
            "Budget is a spending cap, not income—never ask for income. "
            "All amounts are pre-calculated; use them directly. "
            "Give practical advice with whole numbers (e.g., '3 meals' not '2.7 meals'). "
            "Be concise and realistic. "
            "Use markdown for formatting. "
            "(DO NOT USE TITLES OR HEADING, Just Bold text wherever a heading is required). "
            "Always add a newline after a bold heading."
        ),
    }

    context_msg = {
        "role": "user",
        "content": (
            f"User's spending data:\n"
            f"Total spent: ₹{total_spent:,.2f}\n"
            f"Average daily: ₹{avg_daily:,.2f}\n"
            f"Days tracked: {len(daily_spending)}\n\n"
            f"Spending by category:\n{category_summary}"
        ),
    }

    trimmed_history = chat_history[-6:] if chat_history else []

    messages = [system_msg, context_msg] + trimmed_history + [
        {"role": "user", "content": question}
    ]

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.4,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Chatbot error: {e}"