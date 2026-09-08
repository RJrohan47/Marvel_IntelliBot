"""General-purpose tools available when a chat has no active video context."""

from __future__ import annotations

import os

import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool


search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform add, subtract, multiply, or divide on two numbers."""
    operations = {
        "add": lambda: first_num + second_num,
        "subtract": lambda: first_num - second_num,
        "multiply": lambda: first_num * second_num,
        "divide": lambda: first_num / second_num if second_num != 0 else None,
    }
    if operation not in operations:
        return {"error": f"Unsupported operation '{operation}'"}
    if operation == "divide" and second_num == 0:
        return {"error": "Division by zero is not allowed"}
    return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": operations[operation]()}


@tool
def get_weather_info(city: str) -> dict:
    """Fetch current weather data for a city from Weatherstack."""
    api_key = os.getenv("WEATHERSTACK_API_KEY")
    if not api_key:
        return {"error": "Weather service is not configured. Set WEATHERSTACK_API_KEY in .env."}
    try:
        response = requests.get("https://api.weatherstack.com/current", params={"access_key": api_key, "query": city}, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"error": f"Weather service request failed: {exc}"}


GENERAL_TOOLS = [search_tool, calculator, get_weather_info]
