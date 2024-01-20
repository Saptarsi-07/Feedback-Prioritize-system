# from flask import Flask, render_template, request, jsonify
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification

# # app = Flask(__name__)

# # # Load the fine-tuned model and tokenizer
# # model = BertForSequenceClassification.from_pretrained("fine_tune_model")
# # tokenizer = BertTokenizer.from_pretrained("fine_tune_tokenizer")

# # # Define a route for the home page
# # @app.route('/')
# # def home():
# #     return render_template('index.html')

# # # Define a route for making predictions
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     # Get the input text from the form
# #     input_text = request.form.get('input_text')

# #     # Tokenize the input text
# #     tokenized_input = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")

# #     # Make the prediction
# #     model.eval()
# #     with torch.no_grad():
# #         input_ids = tokenized_input["input_ids"]
# #         attention_mask = tokenized_input["attention_mask"]
# #         outputs = model(input_ids, attention_mask=attention_mask)

# #     # Get the predicted label
# #     logits = outputs.logits
# #     predicted_label = torch.argmax(logits, dim=1).item()

# #     return render_template('index.html',predicted_label=predicted_label)

# # if __name__ == '__main__':
# #     app.run(debug=True)





# from flask import Flask, render_template, request

# app = Flask(__name__)

# # Load the fine-tuned model and tokenizer
# model = BertForSequenceClassification.from_pretrained("fine_tune_model")
# tokenizer = BertTokenizer.from_pretrained("fine_tune_tokenizer")


# def predict_sentiment(text):

#     # Tokenize the input text
#     tokenized_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

#     # Make the prediction
#     model.eval()
#     with torch.no_grad():
#         input_ids = tokenized_input["input_ids"]
#         attention_mask = tokenized_input["attention_mask"]
#         outputs = model(input_ids, attention_mask=attention_mask)

#     # Get the predicted label
#     logits = outputs.logits
#     predicted_label = torch.argmax(logits, dim=1).item()

#     return predicted_label




# @app.route('/')
# def index():
#     return render_template('index.html')




# @app.route('/predict', methods=['POST'])
# def home():
#     if request.method == 'POST':
#         user_input = request.form['user_input']
#         predicted_label = predict_sentiment(user_input)
#         return render_template('index.html', predicted_label=predicted_label)
#     else:
#         return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)










from flask import Flask, render_template, request
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("fine_tune_distilbert_model")
tokenizer = DistilBertTokenizer.from_pretrained("fine_tune_distilbert_tokenizer")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        new_input_text = request.form['new_input_text']

        # Tokenize the new input text
        tokenized_input = tokenizer(new_input_text, padding=True, truncation=True, return_tensors="pt")

        # Make the prediction
        model.eval()
        with torch.no_grad():
            input_ids = tokenized_input["input_ids"]
            attention_mask = tokenized_input["attention_mask"]
            outputs = model(input_ids, attention_mask=attention_mask)

        # Get the predicted label
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        # temp = False
        if(predicted_label==1):
            predicted_label="HIGH"
            # temp = True
        else:
            predicted_label="LOW"
            # temp = True



        return render_template('index.html', prediction=predicted_label)
        # if(temp == True):
        #     return render_template('index.html', prediction=predicted_label)
        # else:
        #     return render_template('index.html', prediction="")

if __name__ == '__main__':
    app.run(debug=True)
