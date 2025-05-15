# app.py

import streamlit as st
import openai
import pytesseract
from PIL import Image
import os
import io

 import support_arabic_text

# تفعيل دعم النصوص العربية في جميع المكونات
support_arabic_text(all=True)

# 🟢 التقاط صورة بالكاميرا
def get_ocr_from_camera():
 #st.write("التقاط صورة بواسطة الكاميرا ")
 image_data = st.camera_input(":التقاط صورة بواسطة الكاميرا ")

# 🔵 حفظ الصورة وتحليلها لاحقًا
 if image_data is not None:
    # نحفظ الصورة كملف محلي
    with open("saved_image.jpg", "wb") as f:
        f.write(image_data.getbuffer())
    st.success("✅ تم حفظ الصورة بنجاح باسم saved_image.jpg")
    
    # st.warning("⚠️ لا توجد صورة محفوظة حتى الآن.")
 return image_data
    
#رفع صورة لملصق المنتج
def upload_image_ocr_from_folder():
 uploaded_file = st.file_uploader("📸: ارفع صورة ", type=["png", "jpg", "jpeg"])

 return uploaded_file

    #حويل الصورة إلى نص
def extract_text_from_image(saved_image):
    with st.spinner("🧠 استخراج النص من الصورة..."):
       # image = Image.open(saved_image)
        ingredients_text = pytesseract.image_to_string(saved_image, lang="eng+ara+sve")
        st.text_area("📄 النص المستخرج من الصورة:", value=ingredients_text , height=200)
    return ingredients_text
    
#  تحليل المكونات باستخدام GPT-4
def analyze_ingredients_with_gpt(ingredients_text):
    prompt = f"""
    هل تحتوي قائمة المكونات التالية على أي مكون مشتق من الحشرات؟
    إذا كان نعم، اذكر المكون ووضح مصدره. إذا لا، قل أنها خالية.
    قائمة المكونات:
    {ingredients_text}
    """
    try:
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
        result= response['choices'][0]['message']['content']
        st.success("✅ نتيجة التحليل:")
        st.markdown(result)
    except Exception as e:
         st.error(f"❌ خطأ أثناء الاتصال بـ GPT-4: {e}")
    
# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY") or "ضع_مفتاحك_هنا"

# إذا كنت على Windows:
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="تحليل المكونات الغذائية", page_icon="🐞", layout="centered")

st.title("🐞 تحليل المكونات الغذائية")
st.write("تحقق مما إذا كانت قائمة المكونات تحتوي على مشتقات من الحشرات.")

manual_input = st.text_area("أو أدخل المكونات يدويًا")


# 🟢 التقاط صورة بالكاميرا
saved_image = get_ocr_from_camera()
    
#رفع صورة لملصق المنتج
saved_image = upload_image_ocr_from_folder()
st.write("ارفع صورة لملصق المنتج وسنقوم بتحليل المكونات لمعرفة ما إذا كانت تحتوي على مشتقات من الحشرات.")

if saved_image:
 st.image(saved_image, caption="📷 الصورة التي تم رفعها", use_column_width=True)
else:
 st.warning("⚠️ لا توجد صورة محفوظة حتى الآن.")

  
 
if saved_image:
        ingredients_text = extract_text_from_image(saved_image)
else:
        ingredients_text = manual_input
        
    # تحليل النص باستخدام GPT-4
if st.button("🔍 تحليل النص"):
   
       st.spinner("🤖 تحليل المكونات باستخدام GPT-4...")
    
       analyze_ingredients_with_gpt(ingredients_text)
       
