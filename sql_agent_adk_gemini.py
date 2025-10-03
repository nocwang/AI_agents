import os
import asyncio
import duckdb
from datetime import datetime
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
import nest_asyncio

# Apply nest_asyncio to allow nested event loops (needed in environments like Colab)
nest_asyncio.apply()

# Define whether to use Colab secrets for API key
USE_COLAB = True # Set to False if not running in Colab or using environment variable directly


if USE_COLAB:
    # Get API key from Colab secrets and set as environment variable
    from google.colab import userdata
    GEMINI_API_KEY=userdata.get('Gemini_API_KEY')

    if not GEMINI_API_KEY:
        raise ValueError("Please set the 'Gemini_API_KEY' in Colab secrets.")

    # Set the API key as an environment variable for google.adk to pick up
    os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
else:
    # Check if GOOGLE_API_KEY environment variable is set
    GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("Please set the 'GOOGLE_API_KEY' environment variable.")

# Connect to DuckDB and load CSV
con = duckdb.connect(database=':memory:')
# Use a try-except block for table creation in case it already exists or there are issues
try:
    con.sql("CREATE OR REPLACE TABLE transactions AS SELECT * FROM read_csv_auto('transactions.csv')")
    print("transactions table created/replaced successfully.")
except Exception as e:
    print(f"Error creating transactions table: {e}")


# Define tool to execute SQL
def execute_sql(sql: str) -> dict:
    """Execute a SQL query on the DuckDB database and return the result.

    Args:
        sql (str): The SQL query to execute.

    Returns:
        dict: {"status": "success", "report": result} or {"status": "error", "error_message": error}
    """
    try:
        print(f"Executing SQL query: {sql}")
        result = con.sql(sql).df()
        return {"status": "success", "report": result.to_string()}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

# System instruction for the agent
system_instruction = """You are a SQL agent that queries a DuckDB database with a table named 'transactions'.
Given a user query, generate the appropriate SQL query to answer it.
If you need schema information, query it first using SQL (e.g., DESCRIBE transactions).
Use the execute_sql tool to run your SQL queries and get results.
Respond with the final answer based on the query results."""

# Create the agent
sql_agent = Agent(
    name="sql_agent",
    model="gemini-2.5-flash",  # Or "gemini-2.0-flash"
    description="Agent to generate and execute SQL queries on transactions data.",
    instruction=system_instruction,
    tools=[execute_sql],
)

# Set up runner
runner = InMemoryRunner(
    agent=sql_agent,
    app_name='sql_app',
)

# Modified create_session and run_agent to use await
async def create_session():
    return await runner.session_service.create_session(
        app_name='sql_app', user_id='user'
    )

async def run_agent(session_id: str, query: str):
    content = types.Content(
        role='user', parts=[types.Part.from_text(text=query)]
    )
    print(f"** User: {query}")
    # Use a regular for loop instead of async for
    for event in runner.run(
        user_id='user',
        session_id=session_id,
        new_message=content,
    ):
        if event.content.parts and event.content.parts[0].text:
            print(f"** Agent: {event.content.parts[0].text}")
    print()

# Example usage
if __name__ == "__main__":
    # Use await to call the async functions
    # asyncio.run(create_session()) # Removed as it's within a main execution block
    # asyncio.run(run_agent(session.id, "for every month, every merchant, return the unique amount")) # Removed as it's within a main execution block

    async def main():
        session = await create_session()
        await run_agent(session.id, "for every month, every merchant, return the unique amount")

    # Run the main async function
    asyncio.run(main())