# Import necessary modules
import os
import logging
import re  # Import the regex module for cleaning headlines
import streamlit as st  # Streamlit for the app interface
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from a .env file
load_dotenv()

# Retrieve API keys from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')  # OpenAI API key
google_api_key = os.getenv('GOOGLE_API_KEY')  # Google API key

# Check if API keys are loaded correctly
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

# Initialize the OpenAI language model with the specified model and API key
openai_llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # OpenAI model to use
    openai_api_key=openai_api_key  # API key for authentication
)

# Initialize the Google Generative AI (Gemini) language model with the specified model and API key
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # Google Gemini model to use
    google_api_key=google_api_key  # API key for authentication
)

# Function to generate blog post headlines based on a topic
def generate_headlines(topic, num_headlines=5):
    try:
        # Create a prompt template for generating headlines
        prompt_template = ChatPromptTemplate.from_template(
            f"Generate {num_headlines} engaging blog post headlines about {topic}."
        )
        # Use the pipe operator directly
        chain = prompt_template | openai_llm
        # Run the chain with the given topic
        response = chain.invoke({"topic": topic})
        # Extract the content from the AIMessage object
        response_text = response.content
        # Split the response into individual headlines and clean them
        headlines = response_text.strip().split('\n')
        # Remove any leading numbers or serial prefixes from the headlines
        cleaned_headlines = [re.sub(r'^\d+[\.\)]\s*', '', headline).strip() for headline in headlines]
        return [headline for headline in cleaned_headlines if headline]
    except Exception as e:
        logging.error(f"Error generating headlines: {e}")
        return []

# Function to expand a headline into a blog post
def expand_headline_to_blog_post(headline, length, tone):
    try:
        # Create a prompt template for expanding a headline into a blog post
        prompt_template = ChatPromptTemplate.from_template(
            f"Write a {length}-word blog post based on the following topic: {headline}. "
            f"Use a {tone.lower()} tone. Do not include the headline in the blog post content."
        )
        # Use the pipe operator directly
        chain = prompt_template | gemini_llm
        # Run the chain with the given headline
        response = chain.invoke({"headline": headline})
        # Extract the content from the AIMessage object
        blog_post = response.content
        # Return the cleaned blog post
        return blog_post.strip()
    except Exception as e:
        logging.error(f"Error expanding headline to blog post: {e}")
        return f"Error generating blog post: {e}"

# Streamlit app
def main():
    st.set_page_config(layout="wide")  # Set layout to wide for two panels

    # Left panel for user interaction
    with st.sidebar:
        st.title("Blog Post Generator")
        st.write("Generate blog posts based on your chosen topic.")

        # Input topic
        topic = st.text_input("Enter the topic for blog post generation:")

        # Number of headlines
        num_headlines = st.slider("Number of Headlines to Generate:", min_value=1, max_value=10, value=5)

        # Blog post length
        length = st.selectbox("Select Blog Post Length (in words):", [300, 500, 800])

        # Blog post tone
        tone = st.selectbox(
            "Select the Tone of the Blog Post:",
            ["Formal", "Informal", "Conversational", "Persuasive", "Optimistic"]
        )

        # Generate headlines button
        if st.button("Generate Headlines"):
            if not topic:
                st.error("Please enter a topic.")
            else:
                st.session_state.headlines = generate_headlines(topic, num_headlines)
                if not st.session_state.headlines:
                    st.error("No headlines were generated. Try a different topic.")
                else:
                    st.success("Headlines generated successfully!")

    # Right panel for displaying content
    with st.container():
        st.title("Generated Content")

        # Display generated headlines and allow selection
        if "headlines" in st.session_state and st.session_state.headlines:
            st.write("### Generated Headlines:")
            selected_headlines = st.multiselect(
                "Select the headlines to expand into blog posts:",
                options=st.session_state.headlines,
            )

            # Generate blog posts button
            if st.button("Generate Blog Posts"):
                if not selected_headlines:
                    st.error("Please select at least one headline.")
                else:
                    st.session_state.blog_posts = []
                    for headline in selected_headlines:
                        st.session_state.blog_posts.append({
                            'headline': headline,
                            'blog_post': expand_headline_to_blog_post(headline, length, tone)
                        })
                    st.success("Blog posts generated successfully!")

        # Display blog posts
        if "blog_posts" in st.session_state and st.session_state.blog_posts:
            for post in st.session_state.blog_posts:
                st.subheader(post['headline'])
                st.write(post['blog_post'])

if __name__ == "__main__":
    main()

