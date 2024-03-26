from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import requests
import streamlit as st

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")




load_dotenv(find_dotenv())

#img2text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]['generated_text']
    print(text)
    return text

#img2text("rower.png")

#llm
def generate_story(text):
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    text_input = tokenizer.encode("As a story teller please generate short story based on a simple narrative, the story should be no more than 20 words: {}".format(text),return_tensors="pt")
    text_input = text_input.to("cuda" if next(model.parameters()).is_cuda else "cpu")
    outputs = model.generate(text_input, max_length=100, num_beams=4, early_stopping=True)
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    parts = story.split(':')
    return parts[1].strip()
    #print(story)
 



#text to speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)


# scenario = img2text("rower.png")
# story = generate_story(scenario)
# # print(story)
# text2speech(story)

def main():
    st.set_page_config(page_title="img 2 audio story")
    st.header("Turn image into audio story")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image.",
                 use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio("audio.flac")

if __name__ == '__main__':
    main()