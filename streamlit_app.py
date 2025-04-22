import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# --- UI部分 ---
st.title("MEO向け 関連KWサジェストRAGツール")
st.write("検索クエリとエリアを入力してください。関連KWとその理由を返します。")

query = st.text_input("検索クエリ（例：近くのホテル）")
area = st.selectbox("エリアを選択", ["東京", "大阪", "福岡"])

if st.button("関連KWを検索"):
    # --- ベクトルDBロード（ここでは仮。エリアごとに分ける想定） ---
    db_path = f"faiss_db/{area}.faiss"
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever()
    llm = ChatOpenAI()

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    result = qa_chain.run(query)

    st.subheader("🔍 関連キーワード＆文脈")
    st.write(result)
