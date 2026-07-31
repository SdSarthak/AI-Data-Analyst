"""
Streamlit front end for the data analyst agent.

All logic lives in the ``analyst`` package; this file only handles widgets,
state and rendering.

Run with:
    streamlit run main.py
"""
import pandas as pd
import streamlit as st

from analyst.config import MAX_RESULT_ROWS, load_config
from analyst.errors import AnalystError
from analyst.llm import build_client, generate_sql
from analyst.schema import DEFAULT_TABLE, build_schema_ddl, describe_columns
from analyst.sql import execute_sql

EXAMPLE_QUESTIONS = [
    "What is the average value in column X?",
    "Show me the top 5 rows sorted by column Y",
    "Count the number of rows where column Z is greater than 100",
    "Which category has the highest total revenue?",
    "Group the data by column C and calculate the sum of column D",
]


@st.cache_resource(show_spinner=False)
def get_client():
    """Build the inference client once per session."""
    return build_client(load_config())


@st.cache_data(show_spinner=False)
def read_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded CSV. Cached on content so re-runs do not re-parse."""
    from io import BytesIO

    return pd.read_csv(BytesIO(file_bytes))


def render_results(result: pd.DataFrame) -> None:
    """Show the result table, a download button and an optional chart."""
    st.subheader("Query results")

    if result.empty:
        st.warning("The query returned no rows. Try rephrasing your question.")
        return

    st.dataframe(result, use_container_width=True)
    st.caption(f"{len(result)} row(s), {result.shape[1]} column(s)")

    st.download_button(
        label="Download results as CSV",
        data=result.to_csv(index=False),
        file_name="query_results.csv",
        mime="text/csv",
    )

    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    if not numeric_cols or len(result) < 2:
        return

    st.subheader("Quick visualisation")
    chart_type = st.selectbox("Chart type", ["Bar", "Line", "Scatter"])
    label_cols = [c for c in result.columns if c not in numeric_cols]

    x_axis = st.selectbox("X axis", label_cols + numeric_cols)
    y_axis = st.multiselect(
        "Y axis",
        [c for c in numeric_cols if c != x_axis],
        default=[c for c in numeric_cols if c != x_axis][:1],
    )
    if not y_axis:
        st.info("Pick at least one numeric column for the Y axis.")
        return

    chart_data = result[[x_axis, *y_axis]]
    if chart_type == "Bar":
        st.bar_chart(chart_data, x=x_axis, y=y_axis)
    elif chart_type == "Line":
        st.line_chart(chart_data, x=x_axis, y=y_axis)
    else:
        st.scatter_chart(chart_data, x=x_axis, y=y_axis[0])


def render_analysis(df: pd.DataFrame, question: str) -> None:
    """Generate SQL for the question, run it and render the outcome."""
    schema = build_schema_ddl(df, table_name=DEFAULT_TABLE)

    with st.spinner("Generating SQL..."):
        try:
            sql = generate_sql(question, df=df, schema=schema, client=get_client())
        except AnalystError as exc:
            st.error(str(exc))
            return

    st.subheader("Generated SQL")
    st.code(sql, language="sql")

    with st.spinner("Running the query..."):
        result, error = execute_sql(df, sql, table_name=DEFAULT_TABLE)

    if error:
        st.error(error)
        st.info(
            "The model wrote a query the database rejected. Rephrasing the "
            "question, or naming the exact columns you mean, usually fixes it."
        )
        return

    render_results(result)


def main() -> None:
    st.set_page_config(
        page_title="Data Analyst Agent",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Data Analyst Agent")
    st.write(
        "Upload a CSV and ask questions in plain English. Your question is "
        "translated into SQL, run against an in-memory copy of the file, and "
        "the results are shown below."
    )

    st.sidebar.header("Upload data")
    uploaded_file = st.sidebar.file_uploader("CSV file", type=["csv"])

    config = load_config()
    if not config.api_key:
        st.sidebar.warning(
            "No Hugging Face token detected. Set HF_TOKEN in your environment "
            "or in a .env file before running a query."
        )
    st.sidebar.caption(f"Model: {config.model} via {config.provider}")
    st.sidebar.caption(f"Results are capped at {MAX_RESULT_ROWS} rows.")

    if uploaded_file is None:
        st.info("Upload a CSV file from the sidebar to get started.")
        st.subheader("Example questions")
        for example in EXAMPLE_QUESTIONS:
            st.markdown(f"- {example}")
        return

    try:
        df = read_csv(uploaded_file.getvalue(), uploaded_file.name)
    except Exception as exc:
        st.error(f"Could not read that CSV: {exc}")
        return

    if df.empty or len(df.columns) == 0:
        st.error("That file contains no data.")
        return

    st.subheader("Dataset preview")
    st.dataframe(df.head(), use_container_width=True)

    col1, col2 = st.columns(2)
    col1.info(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    col2.info(f"Table name in SQL: `{DEFAULT_TABLE}`")

    with st.expander("Columns detected"):
        st.text(describe_columns(df))

    question = st.text_area(
        "Ask a question about this data:",
        height=100,
        placeholder="e.g. What is the average order value by region?",
    )

    if st.button("Run analysis", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            render_analysis(df, question)


if __name__ == "__main__":
    main()
