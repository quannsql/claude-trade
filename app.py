import os
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from hyperliquid.info import Info
from hyperliquid.utils import constants
import eth_account
from eth_account.signers.local import LocalAccount

from bot_engine import run_bot_async, logger as bot_logger, PRIVATE_KEY, API_URL

app = FastAPI()

# ---------------------------------------------------------
# Logging setup for WebSockets
# ---------------------------------------------------------
class WebSocketLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.connected_websockets = set()

    def emit(self, record):
        log_entry = self.format(record)
        for ws in list(self.connected_websockets):
            try:
                # We schedule the coroutine to run on the event loop
                asyncio.create_task(ws.send_text(log_entry))
            except Exception:
                self.connected_websockets.remove(ws)

ws_handler = WebSocketLogHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
ws_handler.setFormatter(formatter)
bot_logger.addHandler(ws_handler)

# ---------------------------------------------------------
# Hyperliquid Info for API
# ---------------------------------------------------------
info = Info(API_URL, skip_ws=True)
user_address = None
MAIN_ADDRESS = os.environ.get("HL_MAIN_ADDRESS", "").strip()

if MAIN_ADDRESS:
    user_address = MAIN_ADDRESS
elif PRIVATE_KEY:
    try:
        account: LocalAccount = eth_account.Account.from_key(PRIVATE_KEY)
        user_address = account.address
    except Exception:
        pass

# ---------------------------------------------------------
# Background Task
# ---------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    # Start the bot engine in the background
    asyncio.create_task(run_bot_async())

# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/api/state")
def get_state():
    """Fetch balance and open orders from Hyperliquid API"""
    if not user_address:
        return {"error": "Wallet not configured."}
        
    try:
        # Fetch user state for Margin/Balance
        user_state = info.user_state(user_address)
        
        # Fetch open orders
        open_orders = info.open_orders(user_address)
        
        # Fetch order history (fills)
        fills = info.user_fills(user_address)
        
        return {
            "address": user_address,
            "margin_summary": user_state.get("marginSummary", {}),
            "open_orders": open_orders,
            "fills": fills[:30] if fills else [] # Return top 30 recent fills
        }
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    ws_handler.connected_websockets.add(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_handler.connected_websockets.remove(websocket)

# ---------------------------------------------------------
# Static Files
# ---------------------------------------------------------
os.makedirs("web", exist_ok=True)
app.mount("/static", StaticFiles(directory="web"), name="static")

@app.get("/")
def read_root():
    return FileResponse("web/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
