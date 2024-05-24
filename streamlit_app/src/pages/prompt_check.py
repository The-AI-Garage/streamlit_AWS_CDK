import streamlit as st
from few_shot import build_prompt
import boto3

def main():
    st.set_page_config(
    page_title="Prompt Tester",
    page_icon="🦾",
    )
    st.title("Bienvenido 👋")
    st.markdown(
    """
    Aqui vas a poder observar los prompt templates que se utilizan dependiendo el ticket 
    seleccionado
    """
    )
    bedrock = boto3.client(region_name= 'us-east-1', service_name='bedrock-runtime')
    with st.form('my_form'):
        text = st.text_area('Enter text:', ' ')
        submitted = st.form_submit_button('Submit')
    
        if submitted:
            promptTemplate = build_prompt(text, bedrock)
            prompt = promptTemplate.generate_prompt()
            st.write(prompt.format(text))

if __name__ == '__main__': 
    main()