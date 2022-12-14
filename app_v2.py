import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


def main():
    import streamlit as st
    from transformers import AutoModelWithLMHead, AutoTokenizer

    # Load the BERT model and tokenizer
    model = AutoModelWithLMHead.from_pretrained("bert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Create a text input field
    text_input = st.text_input("Enter some text:", key="text_input")

    # Use the `await st.compute` method to run the BERT model in the background
    # and generate autocomplete suggestions
    autocomplete_results = await st.compute(
        generate_autocomplete_suggestions,
        text_input=text_input,
        model=model,
        tokenizer=tokenizer
    )

    # Display the autocomplete suggestions
    for result in autocomplete_results:
        st.write(result)
    
if __name__ == '__main__':
    main()
