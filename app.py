# app.py

import streamlit as st
import openai
import pytesseract
from PIL import Image
import os

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY") or "ضع_مفتاحك_هنا"

# إذا كنت على Windows:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="تحليل المكونات من صورة", page_icon="📷", layout="centered")

st.title("📷 تحليل المكونات الغذائية من صورة")
st.write("ارفع صورة لملصق المنتج وسنقوم بتحليل المكونات لمعرفة ما إذا كانت تحتوي على مشتقات من الحشرات.")

# رفع الصورة
uploaded_file = st.file_uploader("📸 ارفع صورة المكونات", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="📷 الصورة التي تم رفعها", use_column_width=True)
    
    # تحويل الصورة إلى نص باستخدام OCR
    with st.spinner("🧠 استخراج النص من الصورة..."):
        image = Image.open(uploaded_file)
        extracted_text = pytesseract.image_to_string(image, lang="eng+ara")
        st.text_area("📄 النص المستخرج من الصورة:", value=extracted_text, height=200)

    # تحليل النص باستخدام GPT-4
    if st.button("🔍 تحليل النص"):
        with st.spinner("🤖 تحليل المكونات باستخدام GPT-4..."):
            prompt = f"""
أنت مساعد ذكي مختص في تحليل المكونات الغذائية. 
هل تحتوي القائمة التالية على مكونات مشتقة من الحشرات؟
اشرح بالتفصيل إن وُجدت أي مكونات حشرية (مثل E120، shellac، carmine)، واذكر اسم المادة ومصدرها.
إن لم يوجد، أكد ذلك بوضوح.

النص:
{extracted_text}
"""
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=700
                )
                result = response["choices"][0]["message"]["content"]
                st.success("✅ نتيجة التحليل:")
                st.markdown(result)
            except Exception as e:
                st.error(f"❌ خطأ أثناء الاتصال بـ GPT-4: {e}")
