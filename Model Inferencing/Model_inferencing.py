from flask import Flask, request, jsonify, Response
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import pipeline
import torch


# Model Loding
refined_model = "CodeLlama-7b-Instruct-jForg-enhanced"
#  Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf", trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  # Fix for fp16

#  Flask application
app = Flask(__name__)

# Function for converting the data into YAML and send to user


# Dummy inference function for demonstration purposes
def perform_inference(input_data):
    system_message = "You are a helpful, respectful and honest assistant. Helps user to write jFrog pipline and answers about jFrog pipline process"

    text_gen = pipeline(task="text-generation",
                        model=refined_model,
                        torch_dtype=torch.float16,
                        tokenizer=llama_tokenizer,
                        max_length=200,
                        device_map='auto')

    output = text_gen(f"<s>[INST]<<SYS>>{system_message}<</SYS>>{input_data} [/INST]",
                    do_sample=True,
                    top_k=5,
                    top_p = 0.9,
                    temperature = 0.1,
                    num_return_sequences=1,
                    eos_token_id=llama_tokenizer.eos_token_id,
                    max_length=500) # can increase the length of sequence
    return (output[0]['generated_text'])

# Define a route for inference
@app.route('/inference', methods=['POST'])
def inference():
    # Get the input data from the POST request
    input_data = request.json.get('data')

    if not input_data:
        return jsonify({'error': 'Input data not provided'}), 400

    # Perform inference
    inference_result = perform_inference(input_data)

    # Send the inference result as a response
    response_data = {'result': inference_result}
    return jsonify(response_data)

# Defining the YAML Responce 
@app.route('/data', methods=['GET'])
def get_yaml_data():
    input_data = request.json.get('data')

    if not input_data:
        return jsonify({'error': 'Input data not provided'}), 400
    # Inferencing
    inference_result = perform_inference(input_data)
    
    # Create a Python dictionary (or object) to be converted to YAML


    # Convert the dictionary to YAML format
    yaml_data = yaml.dump(inference_result, default_flow_style=False)

    # Set the content type to 'text/yaml' to indicate a YAML response
    return Response(yaml_data, content_type='text/yaml; charset=utf-8')

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
