# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_id = "meta-llama/Llama-3.2-3B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# messages = [
#     {"role": "user", "content": "Who are you?"}
# ]

# # Format the messages into a prompt using the tokenizer's chat template
# prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# outputs = model.generate(**inputs, max_new_tokens=100)

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))




from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=None, torch_dtype=torch.float32)

# Prepare chat-style prompt
messages = [
    {"role": "user", "content": "Who are you?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Tokenize and move to model device
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response with correct pad token
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id  # ✅ Prevents hanging
    )

# Decode and print output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== Response ===\n", response)






# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# def load_model():
#     try:
#         model_id = "meta-llama/Llama-3.2-3B-Instruct"
#         tokenizer = AutoTokenizer.from_pretrained(model_id)
#         model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             device_map=None,
#             torch_dtype=torch.float32
#         )
#         return tokenizer, model
#     except Exception as e:
#         st.error(f"Model loading failed: {e}")
#         raise e

# tokenizer, model = load_model()

# def get_llama_blog(input_text, no_words, blog_style):
#     try:
#         messages = [
#             {"role": "system", "content": f"You are an expert blog writer for {blog_style}."},
#             {"role": "user", "content": f"Write a blog on '{input_text}' in {no_words} words."}
#         ]
#         prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#         # with torch.no_grad():
#         outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)  # Reduce token count

#         return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
#     except Exception as e:
#         return f"⚠️ Generation failed: {e}"


# st.set_page_config(page_title="Generate Blogs", page_icon='🤖', layout='centered')
# st.header("Generate Blogs 🤖")

# input_text = st.text_input("Enter the Blog Topic")

# col1, col2 = st.columns(2)
# with col1:
#     no_words = st.text_input("No of Words", value="300")
# with col2:
#     blog_style = st.selectbox("Writing the blog for", ["Researchers", "Data Scientist", "Common People"])

# if st.button("Generate"):
#     if input_text and no_words:
#         with st.spinner("Generating blog..."):
#             result = get_llama_blog(input_text, no_words, blog_style)
#             st.write(result)
#     else:
#         st.warning("Please fill in all fields.")
