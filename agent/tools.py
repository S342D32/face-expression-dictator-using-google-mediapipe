import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Laptop stubs — later replace the body with HTTP POST to ESP32
# ----------------------------------------------------------------

@tool
def shake_hand() -> str:
    """
    Make the robot shake hands with the user.
    Call this when the user asks for a handshake or greeting gesture.
    """
    logger.info("[TOOL] shake_hand called")
    # TODO: requests.post("http://esp32.local/shake")
    return "Handshake gesture activated"


@tool
def wave() -> str:
    """
    Make the robot wave at the user.
    Call this when the user asks the robot to wave or say hello.
    """
    logger.info("[TOOL] wave called")
    # TODO: requests.post("http://esp32.local/wave")
    return "Wave gesture activated"


@tool
def blink_led(times: int = 3) -> str:
    """
    Blink the robot's LED a given number of times.
    Call this to signal attention or acknowledgement.
    """
    logger.info("[TOOL] blink_led called | times=%d", times)
    # TODO: requests.post("http://esp32.local/blink", json={"times": times})
    return f"LED blinked {times} times"


# ----------------------------------------------------------------
# Export list — used in nodes.py to bind to LLM
# ----------------------------------------------------------------

ALL_TOOLS = [shake_hand, wave, blink_led]
