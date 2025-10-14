# module for creating a server
# you can host your server on local machine or on a remote server
from mcp.server.fastmcp import FastMCP
from retriever.retrieval import Retriever # own code <- VDB <- ETL <- website scraping
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize MCP server and add tools under MCP
mcp = FastMCP("hybrid_search")

# Load retriever once
retriever_obj = Retriever()
retriever = retriever_obj.load_retriever()

# LangChain DuckDuckGo tool
duckduckgo = DuckDuckGoSearchRun()
# we use MCP to expose tools to web. So you and other people can use your tools
# ---------- Helpers ----------
def format_docs(docs) -> str:
    """Format retriever docs into readable context."""
    if not docs:
        return ""
    formatted_chunks = []
    for d in docs:
        meta = d.metadata or {}
        formatted = (
            f"Title: {meta.get('product_title', 'N/A')}\n"
            f"Price: {meta.get('price', 'N/A')}\n"
            f"Rating: {meta.get('rating', 'N/A')}\n"
            f"Reviews:\n{d.page_content.strip()}"
        )
        formatted_chunks.append(formatted)
    return "\n\n---\n\n".join(formatted_chunks)

# ---------- MCP Tools ----------
@mcp.tool()
async def get_product_info(query: str) -> str:
    """Retrieve product information for a given query from local retriever."""
    try:
        print(f"Server: Starting retrieval for query: {query}")
        docs = retriever.invoke(query)
        print(f"Server: Retrieved {len(docs) if docs else 0} documents")
        context = format_docs(docs)
        if not context.strip():
            print("Server: No context found, returning 'No local results found.'")
            return "No local results found."
        print(f"Server: Returning context of length {len(context)}")
        return context
    except Exception as e:
        print(f"Server: Error in get_product_info: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error retrieving product info: {str(e)}"

@mcp.tool()
async def web_search(query: str) -> str:
    """Search the web using DuckDuckGo if retriever has no results."""
    try:
        return duckduckgo.run(query)
    except Exception as e:
        return f"Error during web search: {str(e)}"

# ---------- Run Server ----------
if __name__ == "__main__":
    print("Starting MCP server with stdio transport...")
    mcp.run(transport="stdio")
    # mcp.run(transport="streamable-http")
