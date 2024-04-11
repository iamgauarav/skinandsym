import streamlit as st

def main():
    st.title("Symptoms Checker and Triage Bot")

    # Add sidebars for common diseases and skin diseases
    selected_common_disease = st.sidebar.button("Common Diseases")
    selected_skin_disease = st.sidebar.button("Skin Diseases")

    # Display content based on sidebar selections
    if selected_common_disease:
        display_common_diseases()
    if selected_skin_disease:
        display_skin_diseases()

def display_common_diseases():
    # Display some common diseases
    st.header("Common Diseases")
    st.write("1. Influenza")
    st.write("2. Common Cold")
    st.write("3. Gastroenteritis")

def display_skin_diseases():
    # Display some skin diseases
    st.header("Skin Diseases")
    st.write("1. Eczema")
    st.write("2. Psoriasis")
    st.write("3. Acne")

if __name__ == "__main__":
    main()
