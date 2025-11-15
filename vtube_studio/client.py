import json
import threading
import time
from pathlib import Path

try:
    import websocket  # type: ignore
except ImportError:  # pragma: no cover
    websocket = None

class VTubeStudioClient:
    def __init__(self, url="ws://localhost:8001"):
        self.url = url
        self.ws = None
        self.connected = False
        self.authenticated = False
        self.auth_token = None
        self.token_path = Path(__file__).with_name("token.json")
        self.request_id_counter = 0
        self.lock = threading.Lock()
        self._load_token()
        self._last_param_log = 0.0
        self._reported_errors = set()
        self._last_error_log = 0.0
        self._pending_request = None
        self._input_params_result = None
        
    def connect(self):
        """Establish WebSocket connection"""
        if websocket is None:
            raise ImportError("websocket-client package is not installed")
        try:
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            wst = threading.Thread(target=self.ws.run_forever)
            wst.daemon = True
            wst.start()
            time.sleep(1)
            return self.connected
        except Exception as e:
            print(f"VTube Studio connection error: {e}")
            return False
    
    def _on_open(self, ws):
        self.connected = True
        print("Connected to VTube Studio")
        if self.auth_token:
            self._authenticate()
        else:
            self._request_token()
    
    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
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
                else:
                    print("VTube Studio: token response missing token field")
            elif message_type == "InputParameterListResponse":
                param_list = self._handle_input_parameter_list(data.get("data", {}))
                if param_list:
                    print(f"VTube Studio available parameters: {', '.join(param_list[:10])}{'...' if len(param_list) > 10 else ''}")
            elif message_type == "APIError":
                error_data = data.get("data", {})
                error_msg = error_data.get("message", "Unknown error")
                now = time.time()
                if "Parameter" in error_msg and "not found" in error_msg:
                    if error_msg not in self._reported_errors:
                        self._reported_errors.add(error_msg)
                        print(f"VTube Studio API error: {error_msg}")
                        print("  (This error will not be shown again)")
                    elif now - self._last_error_log >= 30.0:
                        print(f"VTube Studio: Still receiving parameter errors. Check your model's tracking parameters.")
                        self._last_error_log = now
                else:
                    print(f"VTube Studio API error: {error_msg}")
        except json.JSONDecodeError:
            print(f"VTube Studio: Failed to parse message: {message}")
        except Exception as e:
            print(f"VTube Studio message handler error: {e}")
    
    def _on_error(self, ws, error):
        print(f"VTube Studio error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        self.connected = False
        self.authenticated = False
        print("VTube Studio disconnected")
    
    def _request_token(self):
        """Ask VTube Studio for an authentication token (only required once)."""
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
        """Send authentication request"""
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
                "authenticationToken": self.auth_token  # Include token if re-authenticating
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
        """Send JSON message"""
        if self.ws and self.connected:
            try:
                json_str = json.dumps(message)
                self.ws.send(json_str)
            except websocket.WebSocketConnectionClosedException:
                print("VTube Studio: Connection closed")
                self.connected = False
            except Exception as e:
                print(f"VTube Studio send error: {e}")
    
    def send_parameters(self, parameters):
        """Send input parameter data to VTube Studio"""
        if not (self.connected and self.authenticated):
            return
        
        # Build parameter list with proper format
        param_list = []
        for name, value in parameters.items():
            # Ensure value is a float
            try:
                float_value = float(value)
                param_list.append({
                    "id": name,  # VTube Studio uses "id" not "name"
                    "value": float_value
                })
            except (ValueError, TypeError):
                print(f"Warning: Invalid parameter value for {name}: {value}")
                continue
        
        if not param_list:
            return
        
        with self.lock:
            request_id = f"param_{self.request_id_counter}"
            self.request_id_counter += 1
        
        msg = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": request_id,
            "messageType": "InjectParameterDataRequest",
            "data": {
                "faceFound": True,
                "parameterValues": param_list
            }
        }
        
        self._send(msg)
        now = time.time()
        if now - self._last_param_log >= 5.0:
            print(f"Sent {len(param_list)} params to VTube Studio")
            self._last_param_log = now
    
    def disconnect(self):
        """Disconnect from VTube Studio"""
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                print(f"Error disconnecting from VTube Studio: {e}")
            finally:
                self.connected = False
                self.authenticated = False
                self.auth_token = None

    def trigger_expression(self, expression_file):
        """Trigger an expression in VTube Studio"""
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
        """Deactivate an expression in VTube Studio"""
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
    
    def get_input_parameters(self):
        """Query available input parameters from VTube Studio"""
        if not (self.connected and self.authenticated):
            return None
        
        with self.lock:
            request_id = f"get_params_{self.request_id_counter}"
            self.request_id_counter += 1
        
        msg = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": request_id,
            "messageType": "InputParameterListRequest",
            "data": {}
        }
        
        self._pending_request = request_id
        self._input_params_result = None
        
        self._send(msg)
        time.sleep(0.5)
        return self._input_params_result
    
    def _handle_input_parameter_list(self, data):
        """Handle InputParameterListResponse"""
        parameters = data.get("defaultParameters", []) + data.get("customParameters", [])
        param_names = [p.get("name") for p in parameters if p.get("name")]
        self._input_params_result = param_names
        return param_names