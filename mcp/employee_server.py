from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Employee Server")


# --------------------------------
# Employee data
# --------------------------------

employee_data = {
    "EMP001": {
        "name": "Rahul",
        "department": "Engineering",
        "role": "Software Engineer",
        "leave_balance": 12
    },
    "EMP002": {
        "name": "Priya",
        "department": "HR",
        "role": "HR Specialist",
        "leave_balance": 8
    },
    "EMP003": {
        "name": "Amit",
        "department": "Finance",
        "role": "Financial Analyst",
        "leave_balance": 5
    }
}


# --------------------------------
# MCP TOOL
# --------------------------------

@mcp.tool()
def get_leave_balance(employee_id: str) -> str:
    """
    Get the leave balance of an employee.
    """

    employee = employee_data.get(employee_id)

    if employee is None:
        return "Employee not found."

    return (
        f"{employee['name']} has "
        f"{employee['leave_balance']} casual leaves remaining."
    )


# --------------------------------
# MCP RESOURCE
# --------------------------------

@mcp.resource("employee://{employee_id}")
def get_employee(employee_id: str) -> str:
    """
    Get employee profile information.
    """

    employee = employee_data.get(employee_id)

    if employee is None:
        return "Employee not found."

    return (
        f"Employee ID: {employee_id}\n"
        f"Name: {employee['name']}\n"
        f"Department: {employee['department']}\n"
        f"Role: {employee['role']}"
    )


# --------------------------------
# MCP PROMPT
# --------------------------------

@mcp.prompt()
def employee_summary(employee_id: str) -> str:
    """
    Generate a prompt for creating an employee summary.
    """

    return (
        f"Create a professional summary for employee "
        f"{employee_id}. Include their department, role, "
        f"and relevant employee information."
    )


# --------------------------------
# Start MCP server
# --------------------------------

if __name__ == "__main__":
    mcp.run()