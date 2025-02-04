from flask import Flask, request, jsonify
import torch
import threading
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

# Initialize Flask app
app = Flask(__name__)

model = None
tokenizer = None

lock = threading.Lock()

@app.route('/initialize', methods=['POST'])
def initialize_model():
    """
    Initializes the model for inference. This should be called once at the start.
    Expects model configuration and weights path in the request JSON payload.
    """
    global model, tokenizer
    with lock:
        return jsonify({"message": "Model initialized"}), 200


@app.route('/generate', methods=['POST'])
def generate():
    """
    Perform inference on prompt inputs.
    """
    global model, tokenizer
    if model is None:
        return jsonify({"error": "Model is not initialized"}), 400

@app.route('/free_memory', methods=['GET'])
def free_memory():
    """
    Endpoint to check free memory on the server.
    """
    return jsonify({"message": "Free memory"}), 200
    
@app.route('/shutdown', methods=['POST'])
def shutdown():
    """
    Shutdown the server gracefully.
    """
    with lock:
        def stop_server():
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                raise RuntimeError('Not running with the Werkzeug Server')
            func()

        stop_server()
        return jsonify({"message": "Server shutting down"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
