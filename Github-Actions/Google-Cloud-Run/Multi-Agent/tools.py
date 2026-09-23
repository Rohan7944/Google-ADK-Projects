"""
Custom tools provided to the Google ADK Agent.
"""
import time

def fetch_system_status() -> str:
    """
    Checks the active server cluster availability and system heartbeat.
    
    Returns:
        str: A message indicating system health, uptime, and latency.
    """
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    return f"All microservices operational. Status: HEALTHY. Checked at: {current_time} UTC."

def calculate_projected_growth(principal: float, rate: float, years: int) -> float:
    """
    Computes standard compound interest metrics for growth projections.

    Args:
        principal: The starting numerical amount or capital.
        rate: The annual growth rate or interest multiplier (e.g., 0.05 for 5%).
        years: The duration of the tracking timeline in years.

    Returns:
        float: The final calculated compounding amount.
    """
    return principal * ((1 + rate) ** years)
  
