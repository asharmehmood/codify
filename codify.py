import os
import langchain
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI

langchain.debug = False

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["huggingface"]["huggingface_api_key"]
os.environ["GOOGLE_API_KEY"] = st.secrets["gemini"]["gemini_api_key"]

class code_generator:
    mistral_hf = None
    gemini_model = None
    
    def __init__(self):
        repo_id = "mistralai/Mistral-7B-Instruct-v0.1" #"mistralai/Mistral-7B-v0.1"
        self.mistral_hf = HuggingFaceHub(
            repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_new_tokens": 1024}
        )
        
        safety_sets = [
            {
                "category": "HARM_CATEGORY_DANGEROUS",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
        ]
        self.gemini_model = ChatGoogleGenerativeAI(model="gemini-pro",safety_settings=safety_sets)
 
    def inputs_to_llm(self,memory):
        # memory = ConversationBufferMemory(input_key='user_query', memory_key='chat_history')
        # memory = ConversationBufferWindowMemory(k=5, input_key="user_query", memory_key='chat_history',return_messages=True)
        prompt = """
        You are a python code generator bot, queries will be related to code, kindly answer the user queries.
        one example is given below:
        --------- Example 1 ----------
        user: give me code for subtracting a number from another number.
        you: Certainly! Here's a simple example of subtracting one number from another in Python:
        
        # Subtraction function
        def subtract_numbers(a, b):
            result = a - b
            return result

        # Example usage
        number1 = float(input("Enter the first number: "))
        number2 = float(input("Enter the second number: "))

        result = subtract_numbers(number1, number2)

        print(f"The result of subtracting {{number2}} from {{number1}} is: {{result}}")
        
        This code defines a function subtract_numbers that takes two parameters (a and b) and returns the result of subtracting b from a. The example usage then takes two numbers as input from the user and prints the result of the subtraction.
        --------- Example 1 ----------
        
        Note: if user ask irrelevant question (not related to code) like hi, hello, how are you, reply them gently and say 'I am codify, kindly tell me about your coding problem'
        --------- Example 2 ----------
        user: hey, how are you
        you: hi, I'm codify. how I can assist you?
        --------- Example 2 ----------
        
        Chat History:
        {chat_history}
        
        New query from user is given below, answer it. Format the code in python with new line and tabs.
        user: {user_query}
        you: 
        """
        prompt_template = PromptTemplate(input_variables=["user_query"], template=prompt)
        
        return prompt_template,memory
        
    def code_generator_chain(self,memory_obj,llm_type):
        prompt_template,memory = self.inputs_to_llm(memory_obj)
        
        use_llm = None
        if llm_type =='mistral':
            use_llm = self.mistral_hf
        else:
            use_llm = self.gemini_model
            
        codegen_chain = LLMChain(llm=use_llm, prompt=prompt_template, memory=memory, verbose=True)
        
        return codegen_chain
    

        #       let's say user ask to change something in code:
        # --------- Example 2 ----------
        # user: use meaninful variable name in code
        # you: Certainly! Meaningful variable names enhance code readability. Here's the code with more descriptive variable names:
        
        # # Subtraction function
        # def subtract_numbers(minuend, subtrahend):
        #     difference = minuend - subtrahend
        #     return difference

        # # Example usage
        # first_number = float(input("Enter the minuend: "))
        # second_number = float(input("Enter the subtrahend: "))

        # result = subtract_numbers(first_number, second_number)

        # print(f"The result of subtracting {{second_number}} from {{first_number}} is: {{result}}")
        
        # In this version, I've replaced a and b with minuend and subtrahend, respectively, making the code more self-explanatory. The variable names now convey the purpose of each value in the subtraction operation.
        # --------- Example 2 ---------- 