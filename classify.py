from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import numpy as np
from os import path

BASE_PATH = path.abspath(path.dirname(__file__))
print(f'Current directory {BASE_PATH}')

# model's classes 
class_dict = {0: '/Adult',
              1: '/Arts & Entertainment',
              2: '/Arts & Entertainment/Movies',
              3: '/Arts & Entertainment/Online Media',
              4: '/Arts & Entertainment/TV & Video/Online Video',
              5: '/Business & Industrial',
              6: '/Business & Industrial/Business Services',
              7: '/Computers & Electronics',
              8: '/Finance',
              9: '/Finance/Banking',
              10: '/Finance/Investing',
              11: '/Finance/Investing/Currencies & Foreign Exchange',
              12: '/Games/Computer & Video Games',
              13: '/Hobbies & Leisure',
              14: '/Internet & Telecom/Web Services',
              15: '/Jobs & Education/Education',
              16: '/Online Communities',
              17: '/Shopping',
              18: '/Shopping/Apparel',
              19: '/Shopping/Apparel/Clothing Accessories',
              20: '/Shopping/Apparel/Footwear'
              }

def defining_gpu():
    # Check if GPU is available
    if tf.test.is_gpu_available():
        # Set the GPU device
        device = tf.config.experimental.list_physical_devices('GPU')[0]
        tf.config.experimental.set_visible_devices(device, 'GPU')
        tf.config.experimental.set_memory_growth(device, enable=True)
    else:
        # Use CPU if GPU is not available
        device = '/CPU:0'
    return device

# Device defining with TF
print(f'Script is using {defining_gpu()}')  

# Load the fine-tuned BERT model and tokenizer
print('Model loading...')
model = TFBertForSequenceClassification.from_pretrained(BASE_PATH+'/model_22_06_normalized/')
tokenizer = BertTokenizer.from_pretrained(BASE_PATH+'/tokenizer_22_06_normalized/')


        
def predict_probs(input_text, class_dict, model, tokenizer):
    print('Tokenization of input text...')
    input_ids = tokenizer.encode(input_text, add_special_tokens=True, truncation=True, max_length=128)
    input_ids = np.array(input_ids)[None, :]
    # Perform prediction
    print('Getting predictions...') 
    predictions = model.predict(input_ids)
    # Get the predicted probability for each class
    import tensorflow as tf
    class_probabilities = tf.nn.softmax(predictions.logits, axis=-1)
    # Print the probabilities for each class
    probs = []
    for i, prob in enumerate(class_probabilities[0]):
        class_name = class_dict[int(i)]
        probs.append([class_name, f'{prob:.4f}'])
    # Sort the list in descending order based on probabilities
    sorted_probs = sorted(probs, key=lambda x: float(x[1]), reverse=True)
    top_probs = [' '.join(elem) for elem in sorted_probs[:5]]
    print(f'Entered text: {input_text}')
    return top_probs


input_text = input("Enter text for classification, if you want to finish classification enter 'stop'")

while input_text != 'stop':
  print(predict_probs(input_text, class_dict, model, tokenizer))
  input_text = input("Text for classification: ")