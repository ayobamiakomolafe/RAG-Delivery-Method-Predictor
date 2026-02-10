import streamlit as st
import json
from set_up import *

def main():
    st.set_page_config(page_title="AI Delivery Method Recommender", layout="wide")

    st.title("AI Delivery Method Recommender")
    st.markdown("Answer the questions below to get a recommended AI delivery approach.")

    # Render a styled button tab bar using query params so clicks reload the page
    def render_fields(prefix=""):
        industry = st.selectbox(f"Q1: What is your industry?", [
            "Retail", "Education", "Finance", "Healthcare", "Food & Hospitality",
            "Manufacturing", "Professional Services", "Technology/Software"], key=f"{prefix}industry")

        goal = st.selectbox(f"Q2: Where do you most want AI’s help right now?", [
            "Run business smoothly", "Stay compliant / manage policies",
            "Improve products or services", "Not sure"], key=f"{prefix}goal")

        use_case = st.text_input("Q3: What result would make AI most valuable for you? (e.g., automate support, forecasting, document review)", key=f"{prefix}use_case")

        experience = st.selectbox(f"Q4: How comfortable is your team with AI tools today?", [
            "No AI experience", "Tried simple AI tools", "Use AI sometimes", "Use AI often",
            "AI team expert", "Not sure"], key=f"{prefix}experience")

        data_size = st.selectbox(f"Q5: Do you already keep business data for AI use?", [
            "No data yet", "A handful of records", "Tens of thousands of records",
            "Hundreds of thousands to millions", "10 million+ records", "Not sure"], key=f"{prefix}data_size")

        front_deployment = st.selectbox(f"Q6: For the AI system, where do you want the frontend to run?", [
            "AI company – all parts", "Software tools only", "Major cloud service",
            "Company servers (on-site)", "Not sure – need help"], key=f"{prefix}front_deploy")

        back_deployment = st.selectbox(f"Q7: For the AI engine (Backend), where do you want it to run?", [
            "AI company – all only", "Major cloud service", "Company servers (on-site)", "Not sure – need help"], key=f"{prefix}back_deploy")

        return {
            "industry": industry,
            "goal": goal,
            "use_case": use_case,
            "experience": experience,
            "data_size": data_size,
            "front_deployment": front_deployment,
            "back_deployment": back_deployment
        }

    # Simple in-app tab buttons using session state (no query params, no new browser tab)
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 'rag'

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Predictor With RAG", key="btn_with_rag"):
            st.session_state.active_tab = 'rag'
    with c2:
        if st.button("Predictor Without RAG", key="btn_without_rag"):
            st.session_state.active_tab = 'norag'

    active_tab = st.session_state.active_tab

    if active_tab == "rag":
        st.subheader("With Retrieval-Augmented Generation (RAG)")
        profile = render_fields(prefix="rag_")
       

        if st.button("Generate Recommendation (RAG)", key="rag_submit"):
            with st.spinner("Generating recommendation with RAG context..."):
                response = delivery_method_predictor_1(profile)
                st.subheader("Recommended AI Delivery Method")
                st.success(response)
    else:
        st.subheader("LLM Without RAG")
        profile_no_rag = render_fields(prefix="norag_")
        if st.button("Generate Recommendation", key="norag_submit"):
            with st.spinner("Generating recommendation..."):
                try:
                    response = delivery_method_predictor_2(profile_no_rag)
                except Exception as e:
                    st.error(f"Error running predictor: {e}")
                else:
                    st.subheader("Recommended AI Delivery Method")
                    st.success(response)


if __name__ == '__main__':
    main()