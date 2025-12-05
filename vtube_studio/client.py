import json
import threading
import time
from pathlib import Path
from config import MEDIAPIPE_CAPABILITIES

try:
    import websocket  # type: ignore
except ImportError:  # pragma: no cover
    websocket = None

def load_vts_model_params(path):
    try:
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)
        # Collect 'Input' parameter names, as these are the tracking parameters we inject into
        return {p["Input"] for p in data.get("Parameters", []) if p.get("Input")}
    except Exception as e:
        print(f"[WARN] Failed to load VTS model params from {path}: {e}")
        return set()

def filter_params(params, model_params):
    filtered = {}
    for k, v in params.items():
        if model_params and k not in model_params:
            # Only warn if we actually loaded model params
            print(f"[WARN] VTS model does not have param: {k}")
            continue
        if k not in MEDIAPIPE_CAPABILITIES:
            print(f"[WARN] Mediapipe cannot produce param: {k}")
            continue
        filtered[k] = v
    return filtered

class VTubeStudioClient:
    def __init__(self, url="ws://localhost:8001", model_config_path=None):
        self.url = url
        self.ws = None
        self.connected = False
        self.authenticated = False
        self.auth_token = None
        self.token_path = Path(__file__).with_name("token.json")
        self.request_id_counter = 0
        self.lock = threading.Lock()
        self._load_token()
        
        # Load model parameters if path provided
        self.model_params = set()
        if model_config_path:
            self.model_params = load_vts_model_params(model_config_path)
            print(f"Loaded {len(self.model_params)} parameters from model config.")

    def connect(self):
        """Establish WebSocket connection"""
        if websocket is None:
            raise ImportError("websocket-client package is not installed")
        try:
            # Define callbacks
            def on_open(ws):
                print("\n[CONNECTED → VTS]\n")
                self.connected = True
                if self.auth_token:
                    self._authenticate()
                else:
                    self._request_token()

            def on_message(ws, message):
                print("\n===== INBOUND ← VTS =====")
                try:
                    obj = json.loads(message)
                    print(json.dumps(obj, indent=2))
                    self._handle_message(obj)
                except Exception as e:
                    print(f"Raw message: {message}")
                    print(f"Error parsing JSON: {e}")
                print("=========================\n")

            def on_error(ws, error):
                print("\n[ERROR ← VTS]")
                print(error)
                print("================\n")

            def on_close(ws, code, msg):
                print("\n[DISCONNECTED ← VTS]")
                print("Code:", code, "Msg:", msg)
                print("=======================\n")
                self.connected = False
                self.authenticated = False

            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            wst = threading.Thread(target=self.ws.run_forever)
            wst.daemon = True
            wst.start()
            time.sleep(1)
            return True
        except Exception as e:
            print(f"VTube Studio connection error: {e}")
            return False

    def _handle_message(self, data):
        """Internal message handler for logic (auth, etc.)"""
        try:
            message_type = data.get("messageType")
            
            if message_type == "AuthenticationResponse":
                auth_data = data.get("data", {})
                if auth_data.get("authenticated"):
                    self.authenticated = True
                    self.auth_token = auth_data.get("authenticationToken")
                    if self.auth_token:
                        self._save_token(self.auth_token)
                    print("VTube Studio authenticated!")
                else:
                    error = auth_data.get("reason", "Unknown error")
                    print(f"VTube Studio authentication failed: {error}")
                    self.authenticated = False
                    if "token" in error.lower():
                        self.auth_token = None
                        if self.token_path.exists():
                            try:
                                self.token_path.unlink()
                            except OSError:
                                pass
                        self._request_token()
            elif message_type == "AuthenticationTokenResponse":
                token_data = data.get("data", {})
                token = token_data.get("authenticationToken")
                if token:
                    self.auth_token = token
                    self._save_token(token)
                    print("Received new VTube Studio auth token, authenticating...")
                    self._authenticate()
        except Exception as e:
            print(f"Error in _handle_message: {e}")

    def _request_token(self):
        with self.lock:
            request_id = f"token_{self.request_id_counter}"
            self.request_id_counter += 1

        msg = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": request_id,
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": "VTuber Face Tracker",
                "pluginDeveloper": "VTuber Face Tracker",
                "pluginIcon": None,
            },
        }
        self._send(msg)

    def _authenticate(self):
        with self.lock:
            request_id = f"auth_{self.request_id_counter}"
            self.request_id_counter += 1
        
        msg = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": request_id,
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": "VTuber Face Tracker",
                "pluginDeveloper": "YourName",
                "pluginIcon": None,
                "authenticationToken": self.auth_token
            }
        }
        self._send(msg)

    def _load_token(self):
        try:
            if self.token_path.exists():
                data = json.loads(self.token_path.read_text())
                token = data.get("authenticationToken")
                if token:
                    self.auth_token = token
        except Exception as e:
            print(f"Warning: Failed to load VTube Studio token: {e}")

    def _save_token(self, token):
        try:
            self.token_path.write_text(json.dumps({"authenticationToken": token}))
        except Exception as e:
            print(f"Warning: Failed to save VTube Studio token: {e}")
    
    def _send(self, message):
        """Send JSON message with logging"""
        if self.ws and self.connected:
            try:
                json_str = json.dumps(message)
                
                print("\n===== OUTBOUND → VTS =====")
                print(json.dumps(message, indent=2))
                print("==========================\n")
                
                self.ws.send(json_str)
            except Exception as e:
                print(f"VTube Studio send error: {e}")

    def send_parameters(self, parameters):
        """Send input parameter data to VTube Studio with filtering and debug"""
        if not (self.connected and self.authenticated):
            return
        
        filtered = filter_params(parameters, self.model_params)
        
        if not filtered:
            return

        with self.lock:
            request_id = f"param_{self.request_id_counter}"
            self.request_id_counter += 1
        
        # The "100% Guaranteed" Fix: Explicitly filter bad entries
        final_values = []
        for k, v in filtered.items():
            # Check for None or empty string keys
            if k is None or k == "":
                print(f"[BAD ENTRY DETECTED] Key is None/Empty. Value: {v}")
                continue
            
            entry = {"id": k, "value": float(v)}
            
            # Double check the entry object itself
            if not entry["id"] or entry["id"] is None:
                print("[BAD ENTRY DETECTED]", entry)
                continue
                
            final_values.append(entry)

        packet = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": request_id,
            "messageType": "InjectParameterDataRequest",
            "data": {
                "faceFound": True,
                "mode": "set",
                "parameterValues": final_values
            }
        }
        
        self._send(packet)

    def disconnect(self):
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
            finally:
                self.connected = False
                self.authenticated = False

    def trigger_expression(self, expression_file):
        if not (self.connected and self.authenticated):
            return False
        
        with self.lock:
            request_id = f"expr_{self.request_id_counter}"
            self.request_id_counter += 1
        
        msg = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": request_id,
            "messageType": "ExpressionActivationRequest",
            "data": {
                "expressionFile": expression_file,
                "active": True
            }
        }
        self._send(msg)
        return True
    
    def deactivate_expression(self, expression_file):
        if not (self.connected and self.authenticated):
            return False
        
        with self.lock:
            request_id = f"expr_{self.request_id_counter}"
            self.request_id_counter += 1
        
        msg = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": request_id,
            "messageType": "ExpressionActivationRequest",
            "data": {
                "expressionFile": expression_file,
                "active": False
            }
        }
        self._send(msg)
        return True