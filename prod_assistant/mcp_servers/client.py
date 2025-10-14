import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
# *IMPORTANT*: don't forget to run server in docker container -> add command to Dockerfile
async def main():
    try:
        print("Creating MCP client...")
        client = MultiServerMCPClient({
            # server configuration
            "hybrid_search": {   # server name -> we are doing a retriever and web search
                "command": "python",
                "args": [r"C:\Users\Artur Dragunov\Documents\GIT\ecomm-prod-assistant-portfolio\prod_assistant\mcp_servers\product_search_server.py"],  # relative path
                "transport": "stdio" # we are running on local machine, so we use stdio
            }
        })
        
        # * Testing code *
        # Discover tools
        print("Connecting to MCP server and discovering tools...")
        tools = await client.get_tools()
        print("Available tools:", [t.name for t in tools])

        # Pick tools by name -> the name is the same as the function name in the server
        retriever_tool = next(t for t in tools if t.name == "get_product_info")
        web_tool = next(t for t in tools if t.name == "web_search")

        # --- Step 1: Try retriever first ---
        #query = "Samsung Galaxy S25 price"
        # query = "iPhone 15"
        query = "Dell Laptop?"
        print(f"Querying retriever with: {query}")
        try:
            retriever_result = await asyncio.wait_for(
                retriever_tool.ainvoke({"query": query}), 
                timeout=30.0
            )
            print("\nRetriever Result:\n", retriever_result)
        except asyncio.TimeoutError:
            print("Retriever timed out after 30 seconds")
            retriever_result = "Timeout error"

        # --- Step 2: Fallback to web search if retriever fails ---
        # it can happen that the retriever fails to find the results
        # it is because we use mmr with similarity threshold.
        # So, if the similarity is less than the threshold, the document is not returned
        if not retriever_result.strip() or "No local results found." in retriever_result:
            print("\n No local results, falling back to web search...\n")
            web_result = await web_tool.ainvoke({"query": query})
            print("Web Search Result:\n", web_result)
            
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
