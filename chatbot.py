from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

#define chatbot model
tokenizer = AutoTokenizer.from_pretrained("TheHappyDrone/DialoGPT-medium-Nexus-Nova-turing-v2",padding_side='left')
model = AutoModelForCausalLM.from_pretrained("TheHappyDrone/DialoGPT-medium-Nexus-Nova-turing-v2")

app = Flask(__name__)

#chatbot functionality
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json['user_input']
    bot_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    chat_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.75,top_k=0,top_p=0.95)
    bot_response = tokenizer.decode(chat_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return jsonify({'response': bot_response})

@app.route('/cb')
def index():
    return render_template('chatbot.html')


if __name__ == '__main__':
    app.run()
