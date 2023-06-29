
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory,
                                                  ConversationSummaryMemory,
                                                  ConversationBufferWindowMemory
                                                  )
from langchain.memory import ConversationTokenBufferMemory
from dotenv import load_dotenv
import tiktoken
load_dotenv()

llm = OpenAI(
    temperature=0,
    model_name='text-davinci-003'  # we can also use 'gpt-3.5-turbo'
)

# 1. ConversationBufferMemory
"""
It preserves the raw form of the conversation, allowing the chatbot to
refer back to specific parts accurately. In summary, the ConversationBufferMemory
helps the chatbot remember the conversation history, enhancing the overall
conversational experience.
"""
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

print('1. ConversationBufferMemory:')
conversation.predict(input="My Name is sharath")
conversation.predict(input="I stay in hyderabad, India")
conversation.predict(input="What is my name?")
print(conversation.memory.buffer)

# 2. ConversationBufferWindowMemory
"""
The ConversationBufferWindowMemory is like having a short-term memory
that only keeps track of the most recent interactions. It intentionally
drops the oldest ones to make room for new ones.
"""

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=3)  # number of conversations it remembers
)

print('2. ConversationBufferWindowMemory')
conversation.predict(input="My Name is sharath")
conversation.predict(input="I stay in hyderabad, India")
conversation.predict(input="What is my name?")
print(conversation.memory.buffer)

# 3. ConversationSummaryMemory
"""
With the ConversationBufferMemory, the length of the conversation keeps
increasing, which can become a problem if it becomes too large for our
LLM to handle.
To overcome this, we introduce ConversationSummaryMemory. It keeps a
summary of our past conversation snippets as our history. But how does
it summarize? Here comes the LLM to the rescue! The LLM (Language Model)
helps in condensing or summarizing the conversation, capturing the key information.
So, instead of storing the entire conversation, we store a summarized version. This
helps manage the token count and allows the LLM to process the conversation
effectively. In summary, ConversationSummaryMemory keeps a condensed version of
previous conversations using the power of LLM summarization.
"""

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationSummaryMemory(llm=llm)
)

print('3. ConversationSummaryMemory')
conversation.predict(input="My Name is sharath")
conversation.predict(input="I stay in hyderabad, India")
conversation.predict(input="What is my name?")
print(conversation.memory.buffer)


# 4. ConversationTokenBufferMemory
"""
ConversationTokenBufferMemory is a memory mechanism that stores recent 
interactions in a buffer within the system's memory.
Unlike other methods that rely on the number of interactions, this 
memory system determines when to clear or flush interactions based on 
the length of tokens used.
Tokens are units of text, like words or characters, and the buffer is 
cleared when the token count exceeds a certain threshold. By using 
token length as a criterion, the memory system ensures that the buffer 
remains manageable in terms of memory usage.
This approach helps maintain efficient memory management and enables 
the system to handle conversations of varying lengths effectively.
"""

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationTokenBufferMemory(llm=llm, max_token_limit=20),
)

print('4. ConversationTokenBufferMemory')
conversation.predict(input="My Name is sharath")
conversation.predict(input="I stay in hyderabad")
conversation.predict(input="What is my name?")
print(conversation.memory.buffer)
