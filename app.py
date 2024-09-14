import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load a larger T5 model for better performance
model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Preprocess the uploaded document
def preprocess_document(document):
    return document.read().decode("utf-8")

# Function to generate an answer based on the document and question
def generate_answer(document, question):
    input_text = f"question: {question} context: {document}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=300)  # Increase max_length for more complex answers
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Streamlit app for document-based Q&A
def main():
    st.title("Document-Based Q&A with T5 (Larger Model)")

    # Step 1: Upload the file
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

    if uploaded_file is not None:
        document_content = preprocess_document(uploaded_file)
        
        # Step 2: Input the question
        question = st.text_input("Ask a question about the document:")

        if question:
            # Step 3: Generate the answer using the T5 model
            st.write("Generating response...")
            answer = generate_answer(document_content, question)
            
            # Display the generated answer
            st.write(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()