import os
import asyncio
import numpy
from tqdm import tqdm

import aiogram
from aiogram import Bot, Dispatcher, types

API_TOKEN = None
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

import chromadb
from langchain_community.embeddings import OpenAIEmbeddings

OPENAI_API_KEY = None

client = chromadb.Client()

embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

collection = client.create_collection(name="lectures_collection")

import os

with open("book.txt", "r", encoding="utf-8") as file:
    text = file.read()

texts = text.split('\n\n') 

for i, text in enumerate(texts):
    vector = embedding.embed_documents([text])[0]
    
    collection.add(
        documents=[text],
        embeddings=[vector],
        metadatas=[{"id": i}],
        ids=[str(i)]
    )

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma

llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

vectordb = Chroma(collection_name="lectures_collection", embedding_function=embedding)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

WELCOME_MESSAGE = (
    'Данный бот умеет отвечать на вопросы по книге "Perspectives in Logic" Stephen Cook, Phuong Nguyen". Введите ваш вопрос.'
)

# Обработчик команды /start
@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply(WELCOME_MESSAGE)

@dp.message_handler()
async def answer(message: types.Message):
    await message.answer(qa_chain.run(message.text))

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())