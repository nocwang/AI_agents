import os
import re
import logging
import pathlib
import requests
from typing import Optional
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_env_var(var_name: str) -> str:
    """Retrieve environment variable or raise an exception if not found."""
    value = os.getenv(var_name)
    if not value:
        logger.error(f"Environment variable {var_name} not found")
        raise ValueError(f"Environment variable {var_name} must be set")
    return value

class SQLAgent:
    """A class to manage SQL database interactions using LangChain."""
    
    def __init__(self, db_path: str = "Chinook.db"):
        """Initialize the SQL agent with database and LLM configurations."""
        self.db_path = pathlib.Path(db_path)
        self.llm = self._initialize_llm()
        self.db = self._initialize_database()
        self.schema = self.db.get_table_info()
        self.agent = self._create_agent()

    def _initialize_llm(self) -> Any:
        """Initialize the language model with error handling."""
        try:
            openai_api_key = get_env_var("OPENAI_API_KEY")
            return init_chat_model("openai:gpt-4.1")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise

    def _initialize_database(self) -> SQLDatabase:
        """Download and initialize the SQLite database."""
        try:
            if self.db_path.exists():
                logger.info(f"{self.db_path} already exists, skipping download")
            else:
                logger.info(f"Downloading database from {self._get_db_url()}")
                self._download_database()
            return SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def _get_db_url(self) -> str:
        """Return the database download URL."""
        return "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"

    def _download_database(self) -> None:
        """Download the Chinook database if it doesn't exist."""
        try:
            response = requests.get(self._get_db_url(), timeout=10)
            response.raise_for_status()
            self.db_path.write_bytes(response.content)
            logger.info(f"Database downloaded and saved as {self.db_path}")
        except requests.RequestException as e:
            logger.error(f"Failed to download database: {str(e)}")
            raise

    @staticmethod
    def _safe_sql(query: str) -> str:
        """Validate SQL query for safety and append LIMIT if needed."""
        DENY_RE = re.compile(r"\b(INSERT|UPDATE|DELETE|ALTER|DROP|CREATE|REPLACE|TRUNCATE)\b", re.I)
        HAS_LIMIT_TAIL_RE = re.compile(r"(?is)\blimit\b\s+\d+(\s*,\s*\d+)?\s*;?\s*$")
        
        # Normalize query
        query = query.strip()
        if query.count(";") > 1 or (query.endswith(";") and ";" in query[:-1]):
            return "Error: multiple statements are not allowed."
        query = query.rstrip(";").strip()

        # Enforce read-only
        if not query.lower().startswith("select"):
            return "Error: only SELECT statements are allowed."
        if DENY_RE.search(query):
            return "Error: DML/DDL detected. Only read-only queries are permitted."

        # Append LIMIT if not present
        if not HAS_LIMIT_TAIL_RE.search(query):
            query += " LIMIT 5"
        return query

    @tool
    def execute_sql(self, query: str) -> str:
        """Execute a READ-ONLY SQLite SELECT query and return results.
        
        Args:
            query: A string containing a valid SQL SELECT query
            
        Returns:
            A string with query results or an error message
        """
        try:
            safe_query = self._safe_sql(query)
            if safe_query.startswith("Error:"):
                logger.warning(f"Invalid SQL query: {safe_query}")
                return safe_query
            result = self.db.run(safe_query)
            logger.info(f"Successfully executed query: {safe_query}")
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return f"Error: {e}"

    def _create_agent(self) -> Any:
        """Create the LangChain agent with configured prompt and tools."""
        system_prompt = f"""You are a careful SQLite analyst.

Authoritative schema (do not invent columns/tables):
{self.schema}

Rules:
- Think step-by-step.
- When you need data, call the tool `execute_sql` with ONE SELECT query.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limit to 5 rows unless user explicitly asks otherwise.
- If the tool returns 'Error:', revise the SQL and try again.
- Limit the number of attempts to 5.
- If you are not successful after 5 attempts, return a note to the user.
- Prefer explicit column lists; avoid SELECT *.
"""
        try:
            return create_agent(
                model=self.llm,
                tools=[self.execute_sql],
                prompt=SystemMessage(content=system_prompt),
            )
        except Exception as e:
            logger.error(f"Failed to create agent: {str(e)}")
            raise

    def run_query(self, question: str) -> None:
        """Run a query through the agent and print the results."""
        try:
            for step in self.agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                stream_mode="values",
            ):
                step["messages"][-1].pretty_print()
        except Exception as e:
            logger.error(f"Failed to run query: {str(e)}")
            raise

def main():
    """Main function to run the SQL agent."""
    try:
        # Initialize agent
        agent = SQLAgent()
        
        # Example query
        question = "Which genre on average has the longest tracks?"
        logger.info(f"Running query: {question}")
        agent.run_query(question)
        
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()