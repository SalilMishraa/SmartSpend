import re
import numpy as np
import pandas as pd


def process_transactions(df):
    """
    Parse and normalize raw UPI transaction data.

    Returns:
        expenses_df (pd.DataFrame): cleaned expenses with columns
            ['date', 'amount', 'category']
        dropped_rows (int): number of rows dropped due to invalid date/amount
    """

    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]

    # --- Identify amount column ---
    amount_candidates = [c for c in df.columns if 'amount' in c]
    amount_col = amount_candidates[0] if amount_candidates else None

    # --- Identify date column ---
    date_candidates = [c for c in df.columns if 'date' in c]
    date_col = date_candidates[0] if date_candidates else None

    # --- Identify category / tags column ---
    tags_col = next((c for c in df.columns if 'tags' in c), None)
    if not tags_col:
        tags_col = next((c for c in df.columns if 'category' in c), None)
    if not tags_col:
        tags_col = next(
            (
                c for c in df.columns
                if 'narration' in c or 'note' in c
                or 'description' in c or 'details' in c
            ),
            None
        )

    if not all([amount_col, date_col]):
        raise ValueError("File is missing a required 'Amount' or 'Date' column.")

    # --- Amount parsing ---
    def parse_amount(value):
        if pd.isna(value):
            return np.nan

        s = str(value).strip()
        if not s:
            return np.nan

        # Normalize unicode minus
        s = s.replace('\u2212', '-')

        negative = False
        if s.startswith('(') and s.endswith(')'):
            negative = True
            s = s[1:-1]

        s = re.sub(r'[^0-9\.-]', '', s)

        try:
            num = pd.to_numeric(s, errors='coerce')
        except Exception:
            return np.nan

        if pd.isna(num):
            return np.nan

        return -abs(num) if negative else num

    df['amount'] = df[amount_col].apply(parse_amount)

    # --- Apply DR/CR sign if present ---
    sign_series = pd.Series(1, index=df.index)
    for c in df.columns:
        if df[c].dtype == object:
            col_str = df[c].astype(str).str.lower()
            if (
                col_str.str.contains('dr|debit', regex=True, na=False).any()
                or col_str.str.contains('cr|credit', regex=True, na=False).any()
            ):
                sign_series = np.where(
                    col_str.str.contains('dr|debit', regex=True, na=False),
                    -1,
                    1
                )
                sign_series = pd.Series(sign_series, index=df.index)
                break

    df['amount'] = df['amount'] * sign_series

    # --- Date parsing ---
    if np.issubdtype(df[date_col].dtype, np.number):
        # Excel serial dates
        df['date'] = pd.to_datetime(
            df[date_col],
            unit='d',
            origin='1899-12-30',
            errors='coerce'
        )
    else:
        df['date'] = pd.to_datetime(
            df[date_col],
            errors='coerce',
            dayfirst=True,
        )
        if df['date'].isna().mean() > 0.5:
            df['date'] = pd.to_datetime(
                df[date_col],
                errors='coerce',
                dayfirst=False,
            )

    # --- Drop rows only if BOTH amount and date are invalid ---
    before_len = len(df)
    df.dropna(subset=['amount', 'date'], inplace=True)
    dropped_rows = before_len - len(df)

    # --- Keep only expenses (outflows) ---
    expenses_df = df[df['amount'] < 0].copy()
    expenses_df['amount'] = expenses_df['amount'].abs()

    # --- Category normalization ---
    if tags_col is None:
        expenses_df['category'] = 'Uncategorized'
    else:
        if expenses_df[tags_col].dtype == object:
            extracted = expenses_df[tags_col].astype(str).str.strip()
            simple = extracted.str.split().str[-1]
            expenses_df['category'] = simple.where(
                simple.notna() & (simple != ''),
                extracted
            )
        else:
            expenses_df['category'] = 'Uncategorized'

    # --- Final normalized output ---
    expenses_df = expenses_df[['date', 'amount', 'category']].copy()
    expenses_df.sort_values('date', inplace=True)

    return expenses_df, dropped_rows
