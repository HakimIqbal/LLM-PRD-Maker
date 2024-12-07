import os
import uuid
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv

from helpers import (
    get_or_create_memory,
    load_prompt_from_file,
    create_chain,
    conversation_memories
)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)


@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get session ID or create a new one
        session_id = session.get('session_id', str(uuid.uuid4()))
        
        # Get or create memory for this session
        memory = get_or_create_memory(session_id)
        
        # Load prompt template from file
        prompt_template = load_prompt_from_file()
        
        if not prompt_template:
            return jsonify({"error": "Prompt template could not be loaded"}), 400

        # Get data from the request
        data = request.json or {}

        # Prepare the input data as a dictionary (removing human_input)
        inputs = {
            "overview": data.get("overview", ""),
            "start_date": data.get("start_date", ""),
            "end_date": data.get("end_date", ""),
            "document_version": data.get("document_version", ""),
            "product_name": data.get("product_name", ""),
            "document_owner": data.get("document_owner", ""),
            "developer": data.get("developer", ""),
            "stakeholder": data.get("stakeholder", ""),
            "doc_stage": data.get("doc_stage", ""),
            "created_date": data.get("created_date", "")
        }

        # Create chain with memory and prompt
        chain = create_chain(prompt_template, memory)

        # Run the chain using invoke
        result = chain.invoke(inputs)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)


@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    """Clear memory for the current session."""
    session_id = session.get('session_id')
    if session_id in conversation_memories:
        del conversation_memories[session_id]
        return jsonify({"message": "Memory cleared successfully"})
    return jsonify({"error": "No memory found for the session"}), 404

@app.route('/get_history', methods=['GET'])
def get_history():
    """Retrieve the conversation history for the current session."""
    session_id = session.get('session_id')
    memory = get_or_create_memory(session_id)
    history = memory.get_history()
    return jsonify({"history": history})



if __name__ == '__main__':
    app.run(debug=True)
