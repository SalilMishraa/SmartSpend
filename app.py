import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq

from services.ai import generate_ai_suggestions, get_chatbot_response
from services.parser import process_transactions
from services.anomalies import detect_anomalies
from services.metrics import compute_metrics


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
                    
                    # Step 1: Parse transactions
                    expenses_df, st.session_state.dropped_rows = process_transactions(df)

                    # Step 2: Compute metrics
                    metrics = compute_metrics(expenses_df)

                    # Step 3: Detect anomalies
                    metrics["anomalies"] = detect_anomalies(
                        metrics["daily_spend_df"],
                        metrics["category_spending"],
                        expenses_df
                    )

                    # Step 4: Store in session
                    st.session_state.spending_metrics = metrics
                    
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

        # Header
        st.warning(
            f"⚠️ {len(anomalies)} potential anomal"
            f"{'y' if len(anomalies) == 1 else 'ies'} detected"
        )
        st.caption(
            "These entries significantly exceed typical spending patterns. "
            "Review them to confirm they are expected."
        )

        # Convert anomalies to DataFrame
        anomaly_df = pd.DataFrame(anomalies)

        # Normalize / clean fields for display
        anomaly_df['Date'] = anomaly_df['date'].fillna("—")
        anomaly_df['Category'] = anomaly_df['category']
        anomaly_df['Amount'] = anomaly_df['amount'].apply(
            lambda x: f"₹{x:,.2f}"
        )
        anomaly_df['Type'] = anomaly_df['type'].str.capitalize()
        anomaly_df['Reason'] = anomaly_df['reason']

        # Select & order columns for display
        display_df = anomaly_df[
            ['Type', 'Date', 'Category', 'Amount', 'Reason']
        ].copy()

        # Render table
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