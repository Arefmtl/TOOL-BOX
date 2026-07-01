#!/usr/bin/env python3
"""
Demo script to run the ML Pipeline UI
"""

import streamlit as st
from pipeline_page import render_pipeline_page

def main():
    """Run the pipeline demo"""
    st.set_page_config(
        page_title="TOOL-BOX ML Pipeline",
        page_icon="🧠",
        layout="wide"
    )

    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Render the pipeline
    render_pipeline_page()

    # Footer
    st.markdown("---")
    st.markdown("*TOOL-BOX v2.0 - ML Pipeline Skeleton UI*")

if __name__ == "__main__":
    main()