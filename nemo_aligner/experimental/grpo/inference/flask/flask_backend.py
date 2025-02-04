import torch
from nemo_aligner.experimental.grpo.inference.base import InferenceBackendBase
from nemo_aligner.servers.http_communicator import FlaskCommunicator

class FlaskInferenceBackend(InferenceBackendBase):
    def __init__(self, model_cfg, tokenizer, servers, **kwargs):
        """
        Initialize the backend.
        Args:
            model_cfg: Configuration of the model.
            tokenizer: Tokenizer instance.
            servers: Dictionary containing server names and their info (ip and port).
        """
        self.model_cfg = model_cfg
        self.tokenizer = tokenizer
        self.communicator = FlaskCommunicator(servers)
        self.server_name = list(servers.keys())[0]  # Assume single server for now

        # Check connection to the Flask server
        self._check_server_connection()

    def _check_server_connection(self):
        """Ping the server to ensure it's running."""
        future = self.communicator.send_data_to_server(self.server_name, None)
        try:
            result = self.communicator.get_result(future)
            if result.get("status") != "OK":
                raise RuntimeError("Server health check failed.")
        except Exception as e:
            raise RuntimeError(f"Error connecting to inference server: {str(e)}")

    def refit(self, model):
        """Upload model state to the server."""
        # For now, we assume the model is preloaded on the server.
        # Add weight synchronization logic here when necessary.
        pass

    def generate(self, inputs, use_greedy=False):
        """
        Generate text using the Flask server.
        Args:
            inputs: Tuple (prompt_tokens, prompt_lengths).
        """
        prompt_tokens, prompt_lengths = inputs
        max_length = self.model_cfg.grpo.length_params.get("max_length", 1024)

        payload = {
            "prompt_tokens": prompt_tokens.cpu().tolist(),
            "prompt_lengths": prompt_lengths.cpu().tolist(),
            "max_length": max_length,
        }

        # Send generate request to the server
        future = self.communicator.send_data_to_server(self.server_name, payload)
        result = self.communicator.get_result(future)

        if result:
            return {
                "response_tokens": torch.tensor(result["response_tokens"]).cuda(),
                "response_lengths": torch.tensor(result["response_lengths"]).cuda(),
            }
        else:
            raise RuntimeError("Failed to generate response from server.")

    def free(self):
        """Free resources on the server."""
        # Send shutdown signal to server
        payload = {}
        future = self.communicator.send_data_to_server(self.server_name, payload)
        result = self.communicator.get_result(future)
        if result.get("message", "").lower() != "server shutting down":
            raise RuntimeError("Failed to shut down the server.")
