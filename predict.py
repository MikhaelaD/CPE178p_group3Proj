import numpy as np
import mindspore as ms
from mindspore import Tensor
from PIL import Image
from mindspore import load_checkpoint, load_param_into_net
from scipy.special import softmax
import mindspore.dataset.vision as vision_transforms


# RESNET 50 #
from models_architecture.resnet50_model import resnet50

class ResNet50Predict:
    def __init__(self, ckpt_path="models/model_resnet_1/resnet-ai-1-1_177.ckpt"):
        """
        Initialize the ResNet50Predictor with model architecture and checkpoint.
        """
        self.model = resnet50()
        param_dict = load_checkpoint(ckpt_path)
        load_param_into_net(self.model, param_dict)
        self.model.set_train(False)

    def preprocess_image(self, image_path):
        """
        Preprocess the input image for prediction.
        """
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)
        resize_op = vision_transforms.Resize((2564, 256)) 
        centercrop_op = vision_transforms.CenterCrop(224)
        normalize_op = vision_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


        img = resize_op(img)
        img = centercrop_op(img)
        img = normalize_op(img)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img

    def predict(self, image_path):
        """
        Predict the class of the image using the ResNet-50 model.
        """
        processed_image = self.preprocess_image(image_path)
        input_tensor = Tensor(processed_image, ms.float32)
        output = self.model(input_tensor)

        print(output.asnumpy())
        predicted_class = np.argmax(softmax(output.asnumpy()), axis=1)
        
        return predicted_class


# BERT #
from transformers import BertTokenizerFast, BertForQuestionAnswering, BertModel
import torch
import pandas as pd

from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch

class BERTAnswer:
    def __init__(self, model_dir="D:/_results_3/checkpoint-66"):
        """
        Initialize the BERTAnswer model and tokenizer.
        """
        self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        self.model = BertForQuestionAnswering.from_pretrained(model_dir)
        self.model.eval()

        self.context = """
        The eggs of Ascaris lumbricoides, Capillaria philippinensis, Enterobius vermicularis, and Fasciolopsis buski have distinct microscopic features and cause varying symptoms. Ascaris lumbricoides eggs are oval with a thick, bumpy shell, and infections can lead to abdominal pain or respiratory symptoms, treatable with albendazole. Capillaria philippinensis eggs are peanut-shaped with flattened polar plugs, causing diarrhea and weight loss, also treated with albendazole. Enterobius vermicularis eggs are oval and clear, primarily causing perianal itching, treatable with mebendazole. Fasciolopsis buski eggs are large, oval, and have an operculum, leading to abdominal pain and diarrhea, with praziquantel as the treatment. Hygiene and proper sanitation are key to preventing these infections.
        """
        
    def generate_answer(self, user_prompt, resnet_prediction):
        """
        Generate an answer based on the user prompt and a fixed context paragraph.
        """
        inputs = self.tokenizer(user_prompt, self.context, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)

        answer_tokens = inputs.input_ids[0][start_index:end_index + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        return answer


