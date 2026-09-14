import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():

    client = MultiServerMCPClient(
        {
            "employee_server": {
                "transport": "stdio",
                "command": "python",
                "args": ["mcp/employee_server.py"],
            }
        }
    )

    tools = await client.get_tools()

    print("Available MCP tools:")

    for tool in tools:
        print(tool.name)


if __name__ == "__main__":
    asyncio.run(main())