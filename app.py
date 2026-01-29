import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq

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
        "Use markdown for formatting.\n\n"
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

class SmartSpendApp:
    def __init__(self):
        self.setup_page_config()
        self.setup_session_state()
        self.run_app()

    def setup_page_config(self):
        st.set_page_config(page_title="SmartSpend", page_icon="💸", layout="wide")
        st.markdown("<style>.main-header{font-size:3rem;background:linear-gradient(90deg,#56ab2f 0%,#a8e6cf 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;font-weight:bold;margin-bottom:2rem}</style>", unsafe_allow_html=True)

    def setup_session_state(self):
        defaults = {
            'page': 'upload', 'spending_metrics': {}, 'spending_limit': 5000,
            'chat_history': [], 'ai_suggestions': '', 'dropped_rows': 0
            # --- CHANGE: Removed 'last_api_call_time' as it's no longer needed ---
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def setup_groq(self):
        try:
            if 'groq_client' in st.session_state and st.session_state.groq_client is not None:
                return True
            api_key = st.secrets.get("GROQ_API_KEY")
            if api_key:
                st.session_state.groq_client = Groq(api_key=api_key)
                return True
            return False
        except Exception:
            st.session_state.groq_client = None
            return False

    def run_app(self):
        st.markdown('<h1 class="main-header">💸 SmartSpend</h1>', unsafe_allow_html=True)
        st.markdown("A personalized budgeting tool for college students.")
        st.write("---")
        if st.session_state.page == 'upload':
            self.show_upload_page()
        elif st.session_state.page == 'dashboard':
            self.show_dashboard_page()

    def show_upload_page(self):
        st.subheader("1. Upload Your UPI Statement")
        uploaded_file = st.file_uploader("Upload your UPI statement XLSX", type="xlsx")
        st.subheader("2. Set Your Spending Goal")
        st.session_state.spending_limit = st.number_input("Enter your spending limit for next month (in ₹)", min_value=0, value=st.session_state.spending_limit, step=100)
        st.write("---")

        if uploaded_file and st.button("Analyze Statement", use_container_width=True):
            with st.spinner('Analyzing your statement...'):
                try:
                    xls = pd.ExcelFile(uploaded_file)
                    target_sheet = next((s for s in xls.sheet_names if "payment" in s.lower() or "history" in s.lower()), xls.sheet_names[0])
                    df = pd.read_excel(xls, sheet_name=target_sheet, header=0)
                    
                    st.session_state.spending_metrics, st.session_state.dropped_rows = self.process_transactions(df)
                    
                    # Generate AI suggestions via Groq
                    if self.setup_groq():
                        st.session_state.ai_suggestions = generate_ai_suggestions(
                            st.session_state.spending_metrics,
                            st.session_state.spending_limit
                        )
                    else:
                        st.session_state.ai_suggestions = "AI suggestions are unavailable. Could not connect to the Groq API."

                    st.session_state.page = 'dashboard'
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing file: {e}.")

    def show_dashboard_page(self):
        st.sidebar.header("🧭 Navigation")
        if st.sidebar.button("Go Back to Upload"):
            st.session_state.page = 'upload'
            st.cache_data.clear() # Clear cache when starting over
            st.rerun()

        st.subheader("Your Personalized Dashboard")
        
        if st.session_state.dropped_rows > 0:
            st.info(f"💡 Note: **{st.session_state.dropped_rows} rows** with invalid date or amount formats were ignored in this analysis.")

        st.write("---")
        metrics = st.session_state.spending_metrics
        limit = st.session_state.spending_limit
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Spent", f"₹{metrics.get('total_spent', 0):,.2f}")
        col2.metric("Your Spending Limit", f"₹{limit:,.2f}")
        remaining = max(0, limit - metrics.get('total_spent', 0))
        col3.metric("Remaining Budget", f"₹{remaining:,.2f}", delta=f"₹{limit - metrics.get('total_spent', 0):,.2f}")

        st.write("---")
        st.subheader("Spending Breakdown")
        self.display_spending_charts(metrics.get('category_spending', {}))
        
        st.write("---")
        st.subheader("📊 Spending Concentration Insights")
        self.display_category_insights(metrics)
        
        st.write("---")
        st.subheader("📅 Spending Over Time")
        self.display_time_analysis(metrics)
        
        st.write("---")
        st.subheader("🔍 Unusual Transaction Detection")
        self.display_anomalies(metrics)
        
        st.subheader("💡 AI-Powered Suggestions to Meet Your Goal")
        st.markdown(st.session_state.ai_suggestions)

        st.write("---")
        st.subheader("💬 Ask Your Financial Assistant")
        self.show_chatbot()

    def process_transactions(self, df):
        import re
        import numpy as np

        initial_rows = len(df)
        df.columns = [str(col).strip().lower() for col in df.columns]

        # --- Identify amount column robustly ---
        amount_candidates = [c for c in df.columns if 'amount' in c]
        amount_col = amount_candidates[0] if amount_candidates else None

        # --- Identify date column robustly ---
        date_candidates = [c for c in df.columns if 'date' in c]
        date_col = date_candidates[0] if date_candidates else None

        # --- Identify category/tags column with fallbacks ---
        tags_col = next((c for c in df.columns if 'tags' in c), None)
        if not tags_col:
            tags_col = next((c for c in df.columns if 'category' in c), None)
        if not tags_col:
            tags_col = next((c for c in df.columns if 'narration' in c or 'note' in c or 'description' in c or 'details' in c), None)

        if not all([amount_col, date_col]):
            raise ValueError("File is missing a required 'Amount' or 'Date' column.")

        # --- Clean/parse amount values robustly ---
        def parse_amount(value):
            if pd.isna(value):
                return np.nan
            s = str(value).strip()
            if not s:
                return np.nan
            # Normalize unicode minus to hyphen
            s = s.replace('\u2212', '-')
            # Detect parentheses for negatives
            negative = False
            if s.startswith('(') and s.endswith(')'):
                negative = True
                s = s[1:-1]
            # Remove currency symbols and non-numeric chars except digits, dot, and hyphen
            s = re.sub(r'[^0-9\.-]', '', s)
            # Remove thousand separators like stray multiple hyphens/dots
            # (pandas to_numeric will handle remaining)
            try:
                num = pd.to_numeric(s, errors='coerce')
            except Exception:
                num = np.nan
            if pd.isna(num):
                return np.nan
            return -abs(num) if negative else num

        df['amount'] = df[amount_col].apply(parse_amount)

        # --- Apply sign from DR/CR style columns if present ---
        sign_series = pd.Series(1, index=df.index)
        for c in df.columns:
            if df[c].dtype == object:
                col_str = df[c].astype(str).str.lower()
                if (col_str.str.contains('dr|debit', regex=True, na=False).any() or
                    col_str.str.contains('cr|credit', regex=True, na=False).any()):
                    sign_series = np.where(col_str.str.contains('dr|debit', regex=True, na=False), -1, 1)
                    sign_series = pd.Series(sign_series, index=df.index)
                    break

        df['amount'] = df['amount'] * sign_series

        # --- Robust date parsing ---
        if np.issubdtype(df[date_col].dtype, np.number):
            # Excel serial dates
            df['date'] = pd.to_datetime(df[date_col], unit='d', origin='1899-12-30', errors='coerce')
        else:
            # Try common formats with dayfirst
            df['date'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True, infer_datetime_format=True)
            # If many NaT, try without dayfirst
            if df['date'].isna().mean() > 0.5:
                df['date'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=False, infer_datetime_format=True)

        # --- Drop rows only if both amount and date are missing or invalid ---
        before_len = len(df)
        df.dropna(subset=['amount', 'date'], inplace=True)
        dropped_rows = before_len - len(df)

        # --- Determine expenses (outflows) ---
        expenses_df = df[df['amount'] < 0].copy()
        expenses_df['amount'] = expenses_df['amount'].abs()

        # --- Fallback category handling ---
        if tags_col is None:
            tags_col = 'category_fallback'
            expenses_df[tags_col] = 'Uncategorized'
        else:
            # Extract a concise tag if it's a long narrative; fallback to original if split fails
            if expenses_df[tags_col].dtype == object:
                extracted = expenses_df[tags_col].astype(str).str.strip()
                simple = extracted.str.split().str[-1]
                expenses_df[tags_col] = simple.where(simple.notna() & (simple != ''), extracted)
            else:
                expenses_df[tags_col] = 'Uncategorized'

        total_spent = expenses_df['amount'].sum()
        category_spending = expenses_df.groupby(tags_col)['amount'].sum().sort_values(ascending=False).to_dict()

        # --- TIME-BASED SPENDING ANALYSIS ---
        # Compute daily total spending by grouping expenses by date
        daily_spending = expenses_df.groupby('date')['amount'].sum().sort_index()
        
        # Compute weekly total spending (calendar week with Monday as start)
        # Set the week to start on Monday using freq='W-MON'
        weekly_spending = expenses_df.set_index('date')['amount'].resample('W-MON', label='left').sum()
        
        # Calculate average daily spending across all transaction days
        avg_daily_spending = daily_spending.mean() if not daily_spending.empty else 0
        
        # Find top 3 highest-spending days (date + amount)
        # Sort by amount descending and take top 3
        top_spending_days = daily_spending.sort_values(ascending=False).head(3)
        top_3_days = [
            {'date': date.strftime('%Y-%m-%d'), 'amount': amount}
            for date, amount in top_spending_days.items()
        ] if not top_spending_days.empty else []
        
        # Convert daily_spending Series to list of dicts for storage
        daily_spending_data = [
            {'date': date.strftime('%Y-%m-%d'), 'amount': amount}
            for date, amount in daily_spending.items()
        ] if not daily_spending.empty else []

        # --- ANOMALY DETECTION (HEURISTIC-BASED) ---
        # Detect unusual transactions using simple statistical thresholds
        anomalies = []
        
        if not expenses_df.empty and avg_daily_spending > 0:
            # Calculate category-wise average spending for comparison
            # This helps identify transactions that are unusually high for their category
            category_averages = expenses_df.groupby(tags_col)['amount'].mean()
            
            # Threshold 1: 2× average daily spending (detects globally large transactions)
            # Rationale: A transaction twice the daily average is significantly above normal
            daily_threshold = 2 * avg_daily_spending
            
            for idx, row in expenses_df.iterrows():
                transaction_amount = row['amount']
                transaction_category = row[tags_col]
                transaction_date = row['date']
                reasons = []
                
                # Check if transaction exceeds 2× average daily spending
                if transaction_amount > daily_threshold:
                    reasons.append(f"Amount exceeds 2× avg daily spending (₹{avg_daily_spending:,.2f})")
                
                # Threshold 2: 1.5× category average (detects category-specific outliers)
                # Rationale: 1.5× captures transactions that are moderately high for their category
                # while avoiding false positives from normal variation
                if transaction_category in category_averages:
                    category_avg = category_averages[transaction_category]
                    category_threshold = 1.5 * category_avg
                    
                    if transaction_amount > category_threshold:
                        reasons.append(f"Amount exceeds 1.5× category average (₹{category_avg:,.2f})")
                
                # If any threshold was exceeded, flag as potential anomaly
                if reasons:
                    anomalies.append({
                        'date': transaction_date.strftime('%Y-%m-%d'),
                        'category': transaction_category,
                        'amount': transaction_amount,
                        'reason': ' & '.join(reasons)
                    })
            
            # Sort anomalies by amount descending (highest amounts first)
            anomalies.sort(key=lambda x: x['amount'], reverse=True)

        metrics = {
            'total_spent': total_spent,
            'category_spending': category_spending,
            'daily_spending': daily_spending_data,
            'weekly_spending': weekly_spending.to_dict() if not weekly_spending.empty else {},
            'avg_daily_spending': avg_daily_spending,
            'top_3_days': top_3_days,
            'anomalies': anomalies
        }
        return metrics, dropped_rows

    def display_spending_charts(self, category_spending):
        if not category_spending:
            st.write("No spending data to display.")
            return
        df = pd.DataFrame(list(category_spending.items()), columns=['Category', 'Amount']).sort_values('Amount', ascending=False)
        fig = px.pie(df, values='Amount', names='Category', title='Spending by Category', hole=.3, color_discrete_sequence=px.colors.sequential.Aggrnyl)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    def display_category_insights(self, metrics):
        """Display insights about spending concentration across categories."""
        category_spending = metrics.get('category_spending', {})
        total_spent = metrics.get('total_spent', 0)
        
        # Handle edge case: no spending data
        if not category_spending or total_spent == 0:
            st.info("💡 No spending data available for concentration analysis.")
            return
        
        # Get sorted categories by amount (category_spending is already sorted descending)
        sorted_categories = sorted(category_spending.items(), key=lambda x: x[1], reverse=True)
        
        # Compute top category percentage
        top_category, top_amount = sorted_categories[0]
        top_percentage = (top_amount / total_spent) * 100
        
        # Compute top 3 categories percentage (handle fewer than 3 categories)
        top_3_categories = sorted_categories[:3]
        top_3_total = sum(amount for _, amount in top_3_categories)
        top_3_percentage = (top_3_total / total_spent) * 100
        
        # Display metrics using columns for clean layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Top Category Dominance",
                value=f"{top_percentage:.1f}%",
                delta=f"{top_category}"
            )
        
        with col2:
            top_n = len(top_3_categories)
            st.metric(
                label=f"Top {top_n} Categories Combined",
                value=f"{top_3_percentage:.1f}%",
                delta=f"{top_n} of {len(sorted_categories)} categories"
            )
        
        # Display text insights
        st.info(f"💡 **Top category '{top_category}'** accounts for **{top_percentage:.1f}%** of total spending.")
        
        # Only show top 3 insight if there are at least 2 categories
        if len(sorted_categories) >= 2:
            top_3_names = ', '.join([cat for cat, _ in top_3_categories])
            st.info(f"💡 **Top {len(top_3_categories)} categories** ({top_3_names}) account for **{top_3_percentage:.1f}%** of total spending.")
        
        # Determine and display concentration level with qualitative interpretation
        if top_percentage > 50:
            st.warning("⚠️ **High category concentration** - Consider diversifying spending or review if this category dominance is intentional.")
        elif top_percentage >= 30:
            st.info("📊 **Moderate category concentration** - Spending is reasonably balanced with some focus areas.")
        else:
            st.success("✓ **Well-distributed spending** - Your expenses are spread across multiple categories.")

    def display_time_analysis(self, metrics):
        """Display time-based spending analysis with line chart and key metrics."""
        daily_spending = metrics.get('daily_spending', [])
        
        if not daily_spending:
            st.write("No daily spending data to display.")
            return
        
        # Convert daily spending data to DataFrame for plotting
        # Data is already sorted by date from process_transactions()
        daily_df = pd.DataFrame(daily_spending)
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        
        # Create line chart showing daily spending over time
        fig = px.line(
            daily_df,
            x='date',
            y='amount',
            title='Daily Spending Trend',
            labels={'date': 'Date', 'amount': 'Amount Spent (₹)'},
            markers=True
        )
        fig.update_traces(line_color='#56ab2f', marker=dict(size=8))
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Display key time-based metrics below the chart
        avg_daily = metrics.get('avg_daily_spending', 0)
        top_days = metrics.get('top_3_days', [])
        
        # Show average daily spending
        st.metric("Average Daily Spending", f"₹{avg_daily:,.2f}")
        
        # Display top 3 spending days in columns
        if top_days:
            st.caption("🔝 Top Spending Days")
            cols = st.columns(len(top_days))
            
            for idx, (col, day) in enumerate(zip(cols, top_days), 1):
                with col:
                    st.metric(
                        label=f"#{idx} Highest",
                        value=f"₹{day['amount']:,.2f}",
                        delta=day['date']
                    )

    def display_anomalies(self, metrics):
        """Display potential anomalies detected using heuristic thresholds."""
        anomalies = metrics.get('anomalies', [])
        
        if not anomalies:
            st.success("✓ No unusual transactions detected")
            st.caption("All transactions fall within expected spending patterns.")
            return
        
        # Display count and informational message
        st.warning(f"⚠️ {len(anomalies)} potential anomal{'y' if len(anomalies) == 1 else 'ies'} detected")
        st.caption("These transactions exceed typical spending patterns. Review them to ensure they're expected.")
        
        # Convert anomalies to DataFrame for table display
        anomaly_df = pd.DataFrame(anomalies)
        
        # Format the amount column for display
        anomaly_df['Amount (₹)'] = anomaly_df['amount'].apply(lambda x: f"₹{x:,.2f}")
        
        # Rename columns for better presentation
        display_df = anomaly_df[['date', 'category', 'Amount (₹)', 'reason']].copy()
        display_df.columns = ['Date', 'Category', 'Amount', 'Reason']
        
        # Display as a table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    def show_chatbot(self):
        if not self.setup_groq():
            st.warning("Chatbot is unavailable: Groq API key not configured.")
            return
            
        for msg in st.session_state.chat_history:
            st.chat_message(msg['role']).write(msg["content"])
        
        if prompt := st.chat_input("Ask about your spending..."):
            st.session_state.chat_history.append({'role': 'user', 'content': prompt})
            st.chat_message("user").write(prompt)
            
            with st.spinner("Thinking..."):
                response = get_chatbot_response(
                    prompt,
                    st.session_state.spending_metrics,
                    st.session_state.chat_history
                )
                st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                st.chat_message("assistant").write(response)


if __name__ == "__main__":
    SmartSpendApp()