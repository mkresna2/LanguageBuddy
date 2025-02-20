import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
import json
import os
import random

# Custom CSS for styling
styling = """
    <style>
        .main { background-color: #f0f8ff; padding: 20px; border-radius: 10px; }
        .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; }
        .stTextInput>div>input { border: 2px solid #4682b4; border-radius: 5px; }
        .sidebar .sidebar-content { background-color: #e6e6fa; }
        h1 { color: #2f4f4f; font-family: 'Arial'; }
        h2 { color: #4682b4; }
        .success { color: #006400; font-weight: bold; }
        .error { color: #8b0000; font-weight: bold; }
    </style>
"""

# Tool 1: Load Vocabulary
def load_vocabulary(language="spanish"):
    """Loads Spanish vocabulary with English translations from a file."""
    try:
        with open("spanish-to-english.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return vocab["words"]
    except Exception as e:
        return f"Error loading vocabulary: {str(e)}"

vocab_tool = Tool(
    name="load_vocabulary",
    func=load_vocabulary,
    description="Load Spanish vocabulary with English translations",
    return_direct=True
)

# Tool 2: Generate Sentence
def generate_sentence():
    """Generates a simple Spanish sentence using vocabulary."""
    sentences = [
        {"Spanish": "El gato duerme en la cama.", "English": "The cat sleeps on the bed."},
        {"Spanish": "La niña come una manzana.", "English": "The girl eats an apple."},
        {"Spanish": "El perro juega en el parque.", "English": "The dog plays in the park."},
        {"Spanish": "Mi amigo estudia español.", "English": "My friend studies Spanish."},
        {"Spanish": "Ella lee un libro interesante.", "English": "She reads an interesting book."}
    ]
    return random.choice(sentences)

sentence_tool = Tool(
    name="generate_sentence",
    func=generate_sentence,
    description="Generate a random Spanish sentence with its English translation",
    return_direct=True
)

# Tool 3: Check Grammar
def check_grammar(text):
    """Checks basic grammar in a Spanish sentence."""
    if not text:
        return "Error: No text provided."
    basic_rules = [
        text[0].isupper(),  # Starts with capital
        text.endswith("."),  # Ends with period
        len(text.split()) >= 2  # At least two words
    ]
    return "Correct" if all(basic_rules) else "Please ensure the sentence starts with a capital letter, ends with a period, and contains at least two words."

grammar_tool = Tool(
    name="check_grammar",
    func=check_grammar,
    description="Check if a Spanish sentence follows basic grammar rules",
    return_direct=True
)

# Function to initialize the Groq LLM with user-provided API key
def initialize_llm(api_key):
    if not api_key:
        st.error("Please enter a valid Groq API key in the sidebar.")
        return None
    try:
        return ChatGroq(model="mixtral-8x7b-32768", api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Groq: {str(e)}")
        return None

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Language Learning Buddy (Spanish)",
        page_icon="🎓",
        layout="wide"
    )
    
    # Custom CSS to make the main section wider
    st.markdown("""
        <style>
        .block-container {
            max-width: 1200px !important;
            padding: 2rem;
        }
        .stTextArea textarea {
            font-size: 1rem;
            min-height: 100px;
        }
        .stMarkdown {
            font-size: 1rem;
        }
        div[data-testid="column"] {
            min-width: 400px;
            padding: 1rem;
        }
        div.row-widget.stButton {
            text-align: center;
            margin: 1rem 0;
        }
        .stSelectbox {
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown(styling, unsafe_allow_html=True)
    st.title("Language Learning Buddy (Spanish)")
    st.write("Practice Spanish with vocabulary, sentences, grammar, and topical word generation!")

    # Sidebar with API key input and progress
    with st.sidebar:
        st.header("Settings & Progress")
        api_key = st.text_input("Enter your Groq API Key", type="password", key="api_key_input")
        if "api_key" not in st.session_state:
            st.session_state.api_key = api_key
        else:
            st.session_state.api_key = api_key if api_key else st.session_state.api_key

        st.write("- Select an exercise type or get a random one.")
        st.write("- Answer in the text box and submit.")
        st.write("- Ask follow-ups in the chat section.")

        # Progress bar and score
        st.subheader("Your Progress")
        if "score" not in st.session_state:
            st.session_state.score = {"correct": 0, "total": 0}
        score = st.session_state.score
        progress = score["correct"] / score["total"] if score["total"] > 0 else 0
        st.progress(progress)
        st.write(f"Score: {score['correct']} / {score['total']} "
                 f"({int(progress * 100)}%)")

    # Initialize session state variables
    if "score" not in st.session_state:
        st.session_state.score = {"correct": 0, "total": 0}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_exercise" not in st.session_state:
        st.session_state.current_exercise = None
    if "exercise_type" not in st.session_state:
        st.session_state.exercise_type = None
    if "user_answer" not in st.session_state:
        st.session_state.user_answer = ""
    if "topic_input" not in st.session_state:
        st.session_state.topic_input = ""
    if "api_key" not in st.session_state:
        st.session_state.api_key = None
    if "answer_key" not in st.session_state:
        st.session_state.answer_key = ""
    if "full_exercise" not in st.session_state:
        st.session_state.full_exercise = ""

    # Initialize LLM and agent only if API key is provided
    llm = initialize_llm(st.session_state.api_key)
    if llm is None:
        st.stop()

    # Initialize the agent
    agent = initialize_agent(
        tools=[vocab_tool, sentence_tool, grammar_tool],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            "system_message": (
                "You are a friendly and knowledgeable Spanish language tutor. Your role is to help students learn Spanish "
                "through interactive exercises and provide clear, encouraging feedback. You can:\n"
                "- Create translation exercises between Spanish and English\n"
                "- Generate Spanish sentences for practice\n"
                "- Help with grammar exercises and corrections\n"
                "- Generate topic-specific vocabulary with examples\n\n"
                "Always be encouraging and supportive, focusing on the student's progress and learning journey."
            )
        }
    )

    # Main interaction area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("Exercise Area")
        
        # Exercise type selection
        exercise_option = st.selectbox(
            "Choose Exercise Type",
            ["Random", "Translation", "Sentence Creation", "Grammar Check", "Topical Vocabulary"]
        )
        
        # Topic input and exercise generation
        if exercise_option == "Topical Vocabulary":
            topic = st.text_input("Enter a topic (e.g., 'animals', 'food')", key="topic_input")
            if st.button("Generate Vocabulary", key="topic_button"):
                if not topic:
                    st.error("Please enter a topic for vocabulary generation.")
                    st.stop()
                
                try:
                    with st.spinner("Generating vocabulary..."):
                        # First, get the vocabulary with translations (stored for checking)
                        full_prompt = (
                            f"List 5 basic Spanish vocabulary words about {topic}. Format:\n"
                            f"1. pan - bread\n"
                            f"2. agua - water\n"
                            f"3. leche - milk\n"
                            f"4. arroz - rice\n"
                            f"5. sopa - soup\n\n"
                            f"Replace these examples with different words about {topic}. Keep the same format."
                        )
                        
                        st.session_state.chat_history.append({"role": "user", "content": full_prompt})
                        full_response = agent.run(full_prompt)
                        
                        # Verify we got a response
                        if not full_response or not full_response.strip():
                            raise ValueError("No vocabulary was generated. Please try again.")
                        
                        # Store the full response for answer checking
                        st.session_state.full_exercise = full_response
                        
                        # Create the display version (Spanish only)
                        spanish_words = []
                        for line in full_response.strip().split('\n'):
                            if '-' in line:
                                try:
                                    # Extract the Spanish word (before the dash)
                                    spanish = line.split('-')[0].strip()
                                    # Remove the number if present
                                    if '.' in spanish:
                                        spanish = spanish.split('.')[1].strip()
                                    if spanish:  # Only add non-empty words
                                        spanish_words.append(spanish)
                                except Exception as e:
                                    raise ValueError(f"Error processing vocabulary: {str(e)}")
                        
                        # Verify we got Spanish words
                        if not spanish_words:
                            raise ValueError("Could not extract Spanish words from the response. Please try again.")
                        
                        # Format display response
                        display_response = f"Spanish Vocabulary for '{topic}':\n\n"
                        for i, word in enumerate(spanish_words, 1):
                            display_response += f"{i}. {word}\n"
                        
                        # Store and display the exercise
                        st.session_state.current_exercise = display_response
                        st.session_state.exercise_type = exercise_option
                        st.session_state.user_answer = ""
                        
                        # Display the exercise more prominently
                        st.markdown("---")
                        st.markdown(f"### Vocabulary Exercise: {topic}")
                        st.markdown(f"**{display_response}**")  
                        st.markdown("\nEnter the English translations for these Spanish words.")  
                        st.markdown("---")
                except Exception as e:
                    st.error(f"Error generating vocabulary: {str(e)}")
        else:
            if st.button("New Exercise", key="standard_exercise"):
                try:
                    with st.spinner("Generating exercise..."):
                        if exercise_option == "Random":
                            exercise_prompt = (
                                "As a Spanish language tutor, generate one of these exercises randomly:\n"
                                "1. A Spanish sentence to translate to English\n"
                                "2. A set of Spanish words to translate to English\n"
                                "3. A Spanish sentence to check for grammar\n\n"
                                "Provide clear instructions for the student."
                            )
                        elif exercise_option == "Translation":
                            exercise_prompt = (
                                "As a Spanish language tutor, create a translation exercise.\n"
                                "List 5 Spanish words without their translations.\n"
                                "Include a mix of nouns, verbs, and adjectives.\n"
                                "Format your response as:\n\n"
                                "Translate these Spanish words to English:\n"
                                "1. [Spanish word]\n"
                                "2. [Spanish word]\n"
                                "3. [Spanish word]\n"
                                "4. [Spanish word]\n"
                                "5. [Spanish word]"
                            )
                        elif exercise_option == "Sentence Creation":
                            exercise_prompt = (
                                "As a Spanish language tutor, create a Spanish sentence about a common daily activity.\n"
                                "The sentence should use basic vocabulary and common verb tenses.\n"
                                "Format your response as:\n"
                                "Spanish: [the sentence]\n"
                                "Task: Translate this sentence to English"
                            )
                        elif exercise_option == "Grammar Check":
                            exercise_prompt = (
                                "As a Spanish language tutor, create a Spanish sentence that might have grammar issues.\n"
                                "Ask the student to check if the sentence is grammatically correct and fix any errors.\n"
                                "The sentence should focus on common grammar concepts like verb conjugation or gender agreement."
                            )
                        
                        st.session_state.chat_history.append({"role": "user", "content": exercise_prompt})
                        response = agent.run(exercise_prompt)
                        st.session_state.chat_history.append({"role": "agent", "content": response})
                        st.session_state.current_exercise = response
                        st.session_state.exercise_type = exercise_option
                        st.session_state.user_answer = ""
                        
                        # Display the exercise more prominently
                        st.markdown("---")
                        st.markdown("### New Exercise")
                        st.markdown(f"_{exercise_option}_")
                        st.markdown(f"**{response}**")
                        st.markdown("---")
                except Exception as e:
                    st.error(f"Error generating exercise: {str(e)}")
                    st.write("Debug info:")
                    st.write(f"Exercise type: {exercise_option}")
                    st.write(f"Prompt used: {exercise_prompt}")

        # User input
        if st.session_state.current_exercise:
            user_answer = st.text_input(
                "Your Answer",
                value=st.session_state.answer_key,
                key=f"answer_input_{st.session_state.exercise_type}"
            )

            # Update session state with current answer
            if user_answer:
                st.session_state.user_answer = user_answer

            # Submit answer
            if st.button("Submit Answer", key="submit_button"):
                if not st.session_state.user_answer:
                    st.warning("Please enter your answer before submitting.")
                    st.stop()
                    
                st.write("Checking your answer...")
                
                if st.session_state.exercise_type == "Topical Vocabulary":
                    check_prompt = (
                        f"You are a Spanish language tutor. Check these translations:\n\n"
                        f"Spanish words with correct translations:\n{st.session_state.full_exercise}\n\n"
                        f"Student's translations (in order of the Spanish words):\n{st.session_state.user_answer}\n\n"
                        f"Instructions:\n"
                        f"1. Start with 'Correct!' if all translations are right, or 'Not quite.' if there are mistakes\n"
                        f"2. List any incorrect translations and provide the correct ones\n"
                        f"3. Add a short, encouraging comment\n\n"
                        f"Keep your response brief and clear."
                    )
                elif st.session_state.exercise_type == "Translation":
                    check_prompt = (
                        f"You are a Spanish language tutor evaluating a student's answer. Here are the details:\n\n"
                        f"Exercise: {st.session_state.current_exercise}\n"
                        f"Student's answer: {st.session_state.user_answer}\n"
                        f"Exercise type: {st.session_state.exercise_type}\n\n"
                        f"Provide feedback on:\n"
                        f"1. Whether each translation is correct\n"
                        f"2. Point out any incorrect translations\n"
                        f"3. Provide the correct translations for any mistakes\n\n"
                        f"Format your response as:\n"
                        f"- Start with 'Correct!' if all translations are correct, or 'Not quite.' if there are mistakes\n"
                        f"- List any corrections needed\n"
                        f"- Add an encouraging comment at the end"
                    )
                else:
                    check_prompt = (
                        f"You are a Spanish language tutor evaluating a student's answer. Here are the details:\n\n"
                        f"Exercise: {st.session_state.current_exercise}\n"
                        f"Student's answer: {st.session_state.user_answer}\n"
                        f"Exercise type: {st.session_state.exercise_type}\n\n"
                        f"Provide feedback on:\n"
                        f"1. Whether the answer is correct\n"
                        f"2. Any grammar or translation mistakes\n"
                        f"3. Suggestions for improvement if needed\n\n"
                        f"Be encouraging and supportive in your feedback. Start your response with either 'Correct!' or 'Not quite.' "
                        f"to make it easy to determine if the answer was correct."
                    )
                
                try:
                    with st.spinner("Evaluating your answer..."):
                        st.session_state.chat_history.append({"role": "user", "content": check_prompt})
                        feedback = agent.run(check_prompt)
                        st.session_state.chat_history.append({"role": "agent", "content": feedback})
                        
                        # Update score and show feedback
                        st.session_state.score["total"] += 1
                        if feedback.lower().startswith(("correct", "¡correcto", "great", "excellent", "perfect")):
                            st.session_state.score["correct"] += 1
                            st.success(feedback)
                        else:
                            st.error(feedback)
                        
                        # Clear the answer input
                        st.session_state.answer_key = ""
                        st.session_state.user_answer = ""
                except Exception as e:
                    st.error(f"Error checking answer: {str(e)}")

    with col2:
        st.header("Ask Me Anything")
        
        # Question input with placeholder
        question = st.text_area(
            "Ask a question about Spanish",
            placeholder="e.g., 'What's the past tense of comer?', 'How do I say hello?'",
            height=100
        )
        
        # Center the Ask button
        col2_1, col2_2, col2_3 = st.columns([1, 1, 1])
        with col2_2:
            ask_button = st.button("Ask", type="primary")
        
        # Show response in a clean format
        if ask_button and question:
            with st.spinner("Thinking..."):
                chat_prompt = (
                    f"You are a Spanish language tutor. Answer this question about Spanish:\n{question}\n\n"
                    f"Keep your answer clear and concise. If showing examples, format them clearly."
                )
                
                st.session_state.chat_history.append({"role": "user", "content": question})
                response = agent.run(chat_prompt)
                st.session_state.chat_history.append({"role": "agent", "content": response})
                
                # Display the response in a nice format
                st.markdown("---")
                st.markdown("### Answer")
                st.markdown(response)
                st.markdown("---")

    # Display conversation history in a more readable format
    # if st.session_state.chat_history:
    #     st.subheader("Conversation History")
    #     for msg in st.session_state.chat_history:
    #         role = msg["role"]
    #         content = msg["content"]
    #         if role == "user":
    #             st.markdown(f"🧑 **You:** {content}")
    #         else:
    #             st.markdown(f"🤖 **Tutor:** {content}")

if __name__ == "__main__":
    main()