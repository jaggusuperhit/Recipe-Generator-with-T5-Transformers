import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re


# Load the fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('./fine_tuned_t5_recipe_model')
tokenizer = T5Tokenizer.from_pretrained('./fine_tuned_t5_recipe_model')

# Set the device for model (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Clean Text
# Clean the text: Remove special characters and lowercasing
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    return text

# Function to generate recipe from a prompt
def generate_recipe(prompt, model, tokenizer, max_length=150):
    prompt = clean_text(prompt)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Generate the recipe
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=1)

    return tokenizer.decode(output[0], skip_special_tokens=True)


# Streamlit App Layout
st.title("Recipe Generator with T5 Transformer")
st.subheader("Generate recipes based on your prompt using a fine-tuned T5 model!")

# Input for recipe prompt
prompt = st.text_input("Enter a recipe prompt:", "Generate a vegetarian recipe for dinner with tomatoes and spinach")

if st.button("Generate Recipe"):
    if prompt:
        generated_recipe = generate_recipe(prompt, model, tokenizer)
        st.subheader("Generated Recipe:")
        st.write(generated_recipe)
    else:
        st.warning("Please enter a valid recipe prompt.")
