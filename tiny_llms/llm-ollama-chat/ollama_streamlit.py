import streamlit as st
import ollama
from typing import Generator, List, Dict
from dataclasses import dataclass

@dataclass
class Message:
    role: str
    content: str

class OllamaClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def chat_stream(self, messages: List[Dict[str, str]]) -> Generator:
        try:
            stream = ollama.chat(model=self.model_name, messages=messages, stream=True)
            for chunk in stream:
                if chunk['message']['content']:
                    yield chunk['message']['content']
        except Exception as e:
            yield f"Error: {str(e)}\n\nEnsure Ollama is running: `ollama serve`"

class ChatSession:
    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
    def add_message(self, role: str, content: str) -> None:
        st.session_state.messages.append({"role": role, "content": content})
    
    def get_messages(self) -> List[Dict[str, str]]:
        return st.session_state.messages
    
    def clear(self) -> None:
        st.session_state.messages = []
    
    def render_history(self) -> None:
        for message in self.get_messages():
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

class ChatUI:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OllamaClient(model_name)
        self.session = ChatSession()
        self._configure_page()
    
    def _configure_page(self) -> None:
        st.set_page_config(
            page_title="DeepSeek Chat",
            page_icon="🤖",
            layout="centered"
        )
    
    def _render_header(self) -> None:
        st.title("🤖 DeepSeek-R1 Chat")
        st.caption("Powered by DeepSeek-R1 1.5B")
    
    def _render_sidebar(self) -> None:
        with st.sidebar:
            st.header("Settings")
            
            if st.button("Clear Chat"):
                self.session.clear()
                st.rerun()
            
            st.divider()
            st.subheader("Model Info")
            st.write(f"**Model:** {self.model_name}")
            st.caption("💡 Ask for LaTeX format for math equations")
    
    def _handle_user_input(self, prompt: str) -> None:
        self.session.add_message("user", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            
            for chunk in self.client.chat_stream(self.session.get_messages()):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)
        
        self.session.add_message("assistant", full_response)
    
    def _render_footer(self) -> None:
        st.divider()
        st.caption("Built with Streamlit + Ollama")
    
    def run(self) -> None:
        self._render_header()
        self.session.render_history()
        
        if prompt := st.chat_input("Type your message here..."):
            self._handle_user_input(prompt)
        
        self._render_sidebar()
        self._render_footer()

if __name__ == "__main__":
    app = ChatUI(model_name="deepseek-r1:1.5b")
    app.run()