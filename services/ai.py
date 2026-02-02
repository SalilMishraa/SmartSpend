import streamlit as st
# --- CHANGE: Wrap API calls in Streamlit's caching decorator ---
# This prevents re-running the API call if the input data hasn't changed.
# It's the most effective way to prevent 429 errors in Streamlit.
@st.cache_data
def generate_ai_suggestions(metrics, limit):
    """Generates AI suggestions based on spending metrics."""
    if 'groq_client' not in st.session_state or st.session_state.groq_client is None:
        return "AI suggestions are unavailable. Configure your Groq API key."
    if not metrics.get('category_spending'):
        return "AI suggestions are unavailable. Check your uploaded data."

    # Build enriched prompt with key metrics for better AI context
    category_summary = "\n".join([f"- {category}: ₹{amount:,.2f}" for category, amount in metrics['category_spending'].items()])
    
    # Calculate budget status
    remaining = limit - metrics['total_spent']
    budget_status = f"₹{remaining:,.2f} remaining" if remaining > 0 else f"₹{abs(remaining):,.2f} over budget"
    
    # Get transaction period info
    days_tracked = len(metrics.get('daily_spending', []))
    avg_daily = metrics.get('avg_daily_spending', 0)
    
    # Build prompt with enriched context
    prompt = (
        "You are a friendly financial advisor for a college student in India. "
        "Analyze the following spending data and provide 3-4 concise, actionable tips to help them meet their goal. "
        "Use markdown for formatting. (DO NOT USE TITLES OR HEADING, Just Bold text wherever a heading is required) "
        "Use a newline symbol after a bold heading \n\n"
        f"**Spending Goal:** ₹{limit:,.2f}\n"
        f"**Total Spent:** ₹{metrics['total_spent']:,.2f}\n"
        f"**Budget Status:** {budget_status}\n"
        f"**Average Daily Spending:** ₹{avg_daily:,.2f}\n"
        f"**Days Tracked:** {days_tracked}\n"
    )
    
    # Add top 3 spending days if available
    if metrics.get('top_3_days'):
        prompt += "\n**Top Spending Days:**\n"
        for i, day in enumerate(metrics['top_3_days'], 1):
            prompt += f"{i}. {day['date']}: ₹{day['amount']:,.2f}\n"
    
    # Add anomaly info if present
    anomaly_count = len(metrics.get('anomalies', []))
    if anomaly_count > 0:
        prompt += f"**Unusual Transactions Detected:** {anomaly_count}\n"
    
    prompt += "\n**Spending by Category:**\n" + category_summary
    try:
        response = st.session_state.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are SmartSpend, a budgeting assistant for college students in India. Budget is a spending cap, not income. All totals and amounts are pre-calculated—use them as-is. Give practical advice with whole numbers (e.g., '3 meals' not '2.7 meals'). Be concise and realistic."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, an API error occurred: {e}"

def get_chatbot_response(question, metrics, chat_history):
    """Gets a chatbot response based on user question and financial context."""
    if 'groq_client' not in st.session_state or st.session_state.groq_client is None:
        return "Chatbot is unavailable: Groq API key not configured."
    
    # Build enriched context with key spending metrics
    category_summary = "\n".join([f"- {category}: ₹{amount:,.2f}" for category, amount in metrics['category_spending'].items()])
    
    # Get spending limit from session state
    spending_limit = st.session_state.get('spending_limit', 0)
    total_spent = metrics.get('total_spent', 0)
    avg_daily = metrics.get('avg_daily_spending', 0)
    days_tracked = len(metrics.get('daily_spending', []))
    
    # Build concise, context-aware message list with a small chat cache window
    system_msg = {
        "role": "system",
        "content": "You are SmartSpend, a budgeting assistant for college students in India. Budget is a spending cap, not income—never ask for income. All amounts are pre-calculated; use them directly. Give practical advice with whole, real-world actions. Be concise and precise."
    }
    context_msg = {
        "role": "user",
        "content": (
            f"User's spending data:\n"
            f"Total spent: ₹{total_spent:,.2f}\n"
            f"Budget goal: ₹{spending_limit:,.2f}\n"
            f"Avg daily: ₹{avg_daily:,.2f}\n"
            f"Days tracked: {days_tracked}\n\n"
            f"Spending by category:\n{category_summary}"
        )
    }
    # Take the last 6 turns from history to keep tokens light
    history_msgs = []
    if chat_history:
        trimmed = chat_history[-6:]
        for m in trimmed:
            role = m.get('role', 'user')
            content = m.get('content', '')
            history_msgs.append({"role": role, "content": content})

    user_msg = {"role": "user", "content": question}

    try:
        response = st.session_state.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[system_msg, context_msg, *history_msgs, user_msg],
            temperature=0.4,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, an API error occurred: {e}"