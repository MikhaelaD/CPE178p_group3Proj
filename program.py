import os
import shutil
import tempfile
from PIL import Image
import flet as ft
from predict import ResNet50Predict, BERTAnswer

def main(page: ft.Page):
    # FUNCTIONS #
    def pick_files_result(e: ft.FilePickerResultEvent):
        if e.files:
            global selected_file_path 
            selected_file = e.files[0]
            selected_file_text.value = selected_file.name

            temp_dir = tempfile.gettempdir()
            temp_image_path = os.path.join(temp_dir, selected_file.name)

            shutil.copy(selected_file.path, temp_image_path)

            selected_file_path = temp_image_path
            image_viewer.src = selected_file.path
            image_viewer.update()
        selected_file_text.update()

    def predict(image_path):
        if not os.path.isfile(image_path):
            print(f"Error: The file {image_path} does not exist or is not a valid file.")
            return

        predictor = ResNet50Predict(ckpt_path='models/model_resnet_2/resnet-ai-1-1_177.ckpt')
        class_labels = ["Ascaris lumbricoides", "Capillaria philippinensis", "Enterobius vermicularis", "Fasciolopsis buski"]
        try:
            result = predictor.predict(image_path)
            predicted_class = class_labels[result[0]]
            print(f"Predicted class: {predicted_class}")
            chatbox.controls.append(ft.Text(f"CEPBOT: Predicted class is {predicted_class}"))
            chatbox.update()

            del predictor
            return predicted_class
        except Exception as e:
            print(f"Prediction error: {e}")
            chatbox.controls.append(ft.Text(f"CEPBOT: Error during prediction."))
            chatbox.update()

            del predictor
            return False
        
        
    def on_submit(e):
        global selected_file_path 
        if image_viewer.src != "placeholder.png" and selected_file_path:
            resnet_prediction = predict(selected_file_path)

            if resnet_prediction is not False:
                user_prompt = textprompt.value
                answerer = BERTAnswer()

                answer = answerer.generate_answer(user_prompt, resnet_prediction)
                chatbox.controls.append(ft.Text(f"CEPBOT: {answer}"))
                chatbox.update()

                del answerer
        else:
            selected_file_text.value = "Please upload an image."
            selected_file_text.update()

    # COMPONENTS #
    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    selected_file_text = ft.Text()
    selected_file_path = "" 

    # USER CONTROLS #
    filepicker = ft.ElevatedButton(
        "Pick files",
        icon=ft.icons.UPLOAD_FILE,
        on_click=lambda _: pick_files_dialog.pick_files(allow_multiple=False),
    )
    image_viewer = ft.Image(
        src="placeholder.png",
        width=300,
        height=300,
        fit=ft.ImageFit.CONTAIN,
    )
    chatbox = ft.Column([ft.Text("CEPBOT: Hello! How may I help you concerning parasitic eggs?\n")], scroll=ft.ScrollMode.AUTO)
    textprompt = ft.TextField("", hint_text="Type your response.")
    
    submit_button = ft.ElevatedButton(
        "Submit",
        icon=ft.icons.ARROW_RIGHT,
        on_click=on_submit
    )

    # OVERLAY #
    page.overlay.append(pick_files_dialog)

    # MAIN #
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.add(
        ft.Row(
            [
                ft.Container(ft.Column([image_viewer, selected_file_text, filepicker]), margin=10),
                ft.Container(ft.Column([chatbox, textprompt, submit_button]), margin=10)
            ],
            alignment=ft.MainAxisAlignment.CENTER
        )
    )

ft.app(main)
