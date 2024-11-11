import tkinter as tk
import mindspore
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image as PILImage
import numpy as np
from mindspore import Tensor, load_checkpoint, load_param_into_net
import mindspore.dataset.vision as CV
import os


def resnet50(class_num=4):
    """
    Create ResNet50 model (simplified version for inference)
    """
    from mindspore import nn
    
    class ResidualBlock(nn.Cell):
        expansion = 4
        
        def __init__(self, in_channel, out_channel, stride=1):
            super(ResidualBlock, self).__init__()
            channel = out_channel // self.expansion
            self.conv1 = nn.Conv2d(in_channel, channel, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm2d(channel)
            self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=stride, padding=1, pad_mode='pad')
            self.bn2 = nn.BatchNorm2d(channel)
            self.conv3 = nn.Conv2d(channel, out_channel, kernel_size=1, stride=1)
            self.bn3 = nn.BatchNorm2d(out_channel)
            self.relu = nn.ReLU()
            
            self.down_sample = False
            if stride != 1 or in_channel != out_channel:
                self.down_sample = True
            self.down_sample_layer = None
            
            if self.down_sample:
                self.down_sample_layer = nn.SequentialCell([ 
                    nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_channel)
                ])
    
        def construct(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv3(out)
            out = self.bn3(out)
            
            if self.down_sample:
                identity = self.down_sample_layer(identity)
                
            out = out + identity
            out = self.relu(out)
            return out
    
    class ResNet(nn.Cell):
        def __init__(self, block, layer_nums, in_channels, out_channels, strides, num_classes):
            super(ResNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad')
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
            
            self.layer1 = self._make_layer(block, layer_nums[0], in_channels[0], out_channels[0], strides[0])
            self.layer2 = self._make_layer(block, layer_nums[1], in_channels[1], out_channels[1], strides[1])
            self.layer3 = self._make_layer(block, layer_nums[2], in_channels[2], out_channels[2], strides[2])
            self.layer4 = self._make_layer(block, layer_nums[3], in_channels[3], out_channels[3], strides[3])
            
            self.avgpool = nn.AvgPool2d(7)
            self.flatten = nn.Flatten()
            self.fc = nn.Dense(out_channels[3], num_classes)
        
        def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
            layers = []
            layers.append(block(in_channel, out_channel, stride=stride))
            for _ in range(1, layer_num):
                layers.append(block(out_channel, out_channel, stride=1))
            return nn.SequentialCell(layers)
        
        def construct(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.fc(x)
            return x
    
    return ResNet(ResidualBlock,
                 [3, 4, 6, 3],
                 [64, 256, 512, 1024],
                 [256, 512, 1024, 2048],
                 [1, 2, 2, 2],
                 class_num)


# Parasite Classifier (MindSpore)
class ParasiteClassifier:
    def __init__(self, model_path):
        # Initialize the ResNet50 model
        self.net = resnet50(class_num=4)
        
        # Load the trained parameters with debug info
        print("Loading model parameters...")
        param_dict = load_checkpoint(model_path)
        print(f"Number of parameters loaded: {len(param_dict)}")
        
        # Check if end_point parameters need adjustment
        if "end_point.weight" in param_dict and param_dict["end_point.weight"].shape[0] != 4:
            print("Adjusting end_point layer parameters...")
            param_dict["end_point.weight"] = param_dict["end_point.weight"][:4, :]
            param_dict["end_point.bias"] = param_dict["end_point.bias"][:4]
        
        # Load parameters into the network
        load_param_into_net(self.net, param_dict)
        self.net.set_train(False)
        print("Model loaded successfully")
        
        # Class mapping
        self.classes = {
            0: 'Ascaris lumbricoides',
            1: 'Capillaria philippinensis',
            2: 'Enterobius vermicularis',
            3: 'Fasciolopsis buski'
        }
        
    def preprocess_image(self, image_path):
        # Open and preprocess the image
        image = PILImage.open(image_path)
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert image to numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Resize
        resize_op = CV.Resize((256, 256))
        image_array = resize_op(image_array)
        
        # Center crop
        crop_op = CV.CenterCrop(224)
        image_array = crop_op(image_array)
        
        # Normalize
        normalize_op = CV.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                  std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        image_array = normalize_op(image_array)
        
        # Convert to CHW format
        hwc2chw_op = CV.HWC2CHW()
        image_array = hwc2chw_op(image_array)
        
        # Add batch dimension
        image_tensor = Tensor(image_array.reshape((1, 3, 224, 224)), mindspore.float32)
        return image_tensor
    
    def classify_image(self, image_path):
        try:
            # Preprocess the image
            image_tensor = self.preprocess_image(image_path)
            
            # Run inference
            output = self.net(image_tensor)
            
            # Apply softmax
            probabilities = mindspore.ops.Softmax()(output)
            prediction = probabilities.asnumpy()
            
            # Get prediction
            class_idx = np.argmax(prediction)
            confidence = prediction[0][class_idx] * 100
            
            # Get class name
            class_name = self.classes[class_idx]
            
            return True, class_name, confidence
            
        except Exception as e:
            print(f"Error during classification: {str(e)}")
            return False, str(e), 0


# Parasitic Egg Q&A System (GPT-2)
class GPT2QASystem:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        
        # Add special tokens for Q&A format
        special_tokens = {
            'pad_token': '[PAD]',
            'sep_token': '[SEP]',
            'bos_token': '[BOS]',
            'eos_token': '[EOS]'
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Updated knowledge base for the new parasite classes
        self.knowledge_base = {
            "egg_types": {
                "ascaris_lumbricoides": "Ascaris lumbricoides eggs are oval-shaped, measuring 45-75 μm × 35-50 μm, with a thick shell.",
                "capillaria_philippinensis": "Capillaria philippinensis eggs are oval-shaped, measuring 50-60 μm × 30-40 μm, with a smooth shell.",
                "enterobius_vermicularis": "Enterobius vermicularis eggs are ovoid, measuring 50 μm × 25 μm, with a clear flattened side.",
                "fasciolopsis_buski": "Fasciolopsis buski eggs are operculate, measuring 130-150 μm × 60-80 μm, with a yellow-brown color."
            }
        }
    
    def answer_question(self, parasite_class, question):
        # Get parasite knowledge from the knowledge base
        class_name = parasite_class.lower().replace(" ", "_")
        knowledge = self.knowledge_base["egg_types"].get(class_name, "I don't have specific knowledge about this parasite egg.")

        # Format the prompt with knowledge base context
        prompt = f"Parasite: {parasite_class}\nEgg Characteristics: {knowledge}\nQuestion: {question}\nAnswer:"

        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate response using GPT-2
        response = self.model.generate(inputs, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
        output = self.tokenizer.decode(response[0], skip_special_tokens=True)
        
        # Extract the answer part from the response
        answer = output.split('Answer:')[-1].strip()
        return answer


# GUI Application
class ParasiteClassifierGUI:
    def __init__(self):
        self.classifier = ParasiteClassifier(model_path=r"C:\Users\emjhey\OneDrive\Desktop\1st Term\178P\proj\model_resnet_1\resnet-ai-1_final.ckpt")
        self.qa_system = GPT2QASystem()
        
        self.window = tk.Tk()
        self.window.title("Parasitic Egg Detection System")
        
        # Setup GUI elements
        self.setup_gui()
    
    def setup_gui(self):
        # Create a frame for the left side (image display)
        self.left_frame = tk.Frame(self.window)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Create a frame for the right side (question-answering chat)
        self.right_frame = tk.Frame(self.window)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Image section in the left frame
        self.image_label = tk.Label(self.left_frame, text="Select an Image", font=("Arial", 14))
        self.image_label.grid(row=0, column=0, padx=10, pady=10)
        
        self.select_button = tk.Button(self.left_frame, text="Select Image", command=self.select_image)
        self.select_button.grid(row=1, column=0, padx=10, pady=10)
        
        self.displayed_image = tk.Label(self.left_frame)  # Placeholder for the image
        self.displayed_image.grid(row=2, column=0, padx=10, pady=10)

        # Chat history section in the right frame
        self.chat_history = tk.Text(self.right_frame, width=50, height=15, wrap=tk.WORD, font=("Arial", 12), state=tk.DISABLED)
        self.chat_history.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # Question and answer section in the right frame
        self.question_label = tk.Label(self.right_frame, text="Ask a Question about the Parasite", font=("Arial", 14))
        self.question_label.grid(row=1, column=0, padx=10, pady=10)
        
        self.question_entry = tk.Entry(self.right_frame, font=("Arial", 14), width=30)
        self.question_entry.grid(row=1, column=1, padx=10, pady=10)
        
        self.ask_button = tk.Button(self.right_frame, text="Ask Question", command=self.ask_question)
        self.ask_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
    
    def select_image(self):
        image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if image_path:
            result, class_name, confidence = self.classifier.classify_image(image_path)
            if result:
                self.image_path = image_path
                # Display the selected image
                img = Image.open(image_path)
                img = img.resize((200, 200))  # Resize to fit in the window
                img_tk = ImageTk.PhotoImage(img)
                self.displayed_image.config(image=img_tk)
                self.displayed_image.image = img_tk  # Keep a reference to avoid garbage collection
                
                # Update the chat history with the image classification result
                self.update_chat_history(f"System: Class: {class_name}\nConfidence: {confidence:.2f}%")
    
    def ask_question(self):
        question = self.question_entry.get()
        if hasattr(self, 'image_path') and self.image_path:
            # Update the chat history with the user's question
            self.update_chat_history(f"You: {question}")
            
            # Get the predicted class
            result, class_name, _ = self.classifier.classify_image(self.image_path)
            if result:
                answer = self.qa_system.answer_question(class_name, question)
                
                # Update the chat history with the bot's answer
                self.update_chat_history(f"System: {answer}")
        else:
            messagebox.showerror("Error", "Please select an image first!")
    
    def update_chat_history(self, message):
        self.chat_history.config(state=tk.NORMAL)  # Enable editing to update the text
        self.chat_history.insert(tk.END, message + "\n\n")  # Insert message
        self.chat_history.config(state=tk.DISABLED)  # Disable editing after updating
        self.chat_history.yview(tk.END)  # Scroll to the bottom to show the latest message
    
    def run(self):
        self.window.mainloop()


# Run the application
if __name__ == "__main__":
    gui = ParasiteClassifierGUI()
    gui.run()
