"""
P2-T8: Flask HTTP Trigger
Exposes the Orchestrator via HTTP so n8n can trigger the pipeline on schedule.
"""
import logging
from pathlib import Path
import sys

from flask import Flask, jsonify

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

app = Flask(__name__)
logger = logging.getLogger(__name__)


@app.route('/run-pipeline', methods=['POST'])
def run_pipeline():
    """Trigger the daily pipeline via HTTP POST."""
    from agents.agent_orchestrator import AgentOrchestrator

    logger.info("Pipeline triggered via HTTP POST")
    orchestrator = AgentOrchestrator()
    result = orchestrator.run_daily_pipeline()

    if result['status'] == 'ok':
        return jsonify(result), 200
    else:
        return jsonify(result), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'running'}), 200


@app.route('/train', methods=['POST'])
def initial_training():
    """Trigger initial model training."""
    from agents.agent_orchestrator import AgentOrchestrator

    logger.info("Initial training triggered via HTTP POST")
    orchestrator = AgentOrchestrator()
    result = orchestrator.run_initial_training()

    if result['status'] == 'ok':
        return jsonify(result), 200
    else:
        return jsonify(result), 500


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=False)
