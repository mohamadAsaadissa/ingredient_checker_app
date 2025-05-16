# app.py

import streamlit as st
import openai
import pytesseract
from PIL import Image
import os
import io

from arabic_support import support_arabic_text

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
# if uploaded_file:
   # saved_image = st.image(uploaded_file, caption="📷 الصورة التي تم رفعها", use_column_width=True)

 return uploaded_file

    #حويل الصورة إلى نص
def extract_text_from_image(saved_image):
    with st.spinner("🧠 استخراج النص من الصورة..."):
       # image = Image.open(saved_image)
        ingredients_text = pytesseract.image_to_string(saved_image, lang="eng+ara+sve")
        st.text_area("📄 النص المستخرج من الصورة:", value=ingredients_text , height=200)
    return ingredients_text
    
#  تحليل المكونات باستخدام GPT-4
#def analyze_ingredients_with_gpt(ingredients_text):

    
    # دالة لتبديل حالة العرض
def toggle_message_input():
    st.session_state.show_message_input = not st.session_state.show_message_input

def toggle_message_camera():
    st.session_state.show_message_camera = not st.session_state.show_message_camera

def toggle_message_upload():
    st.session_state.show_message_upload = not st.session_state.show_message_upload


# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY") or "ضع_مفتاحك_هنا"

# إذا كنت على Windows:
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# تفعيل دعم النصوص العربية في جميع المكونات

st.set_page_config(page_title="تحليل المكونات الغذائية", page_icon="🐞", layout="centered")
support_arabic_text(all=True)
# التحقق من وجود المفتاح في حالة الجلسة، وإذا لم يكن موجودًا، يتم تهيئته
if 'show_message_input' not in st.session_state:
    st.session_state.show_message_input = True
if 'show_message_camera' not in st.session_state:
    st.session_state.show_message_camera = False
if 'show_message_upload' not in st.session_state:
    st.session_state.show_message_upload = False
# إنشاء ثلاثة أعمدة بنسبة عرض متساوية
col1, col2, col3 = st.columns([1, 3, 1])

saved_image =""
ingredients_text=""
with col2:
 st.title("🐞 تحليل المكونات الغذائية")

st.write("تحقق مما إذا كانت قائمة المكونات تحتوي على مشتقات من الحشرات.")


    # زر لتبديل عرض الرسالة
st.button("✍️ أدخل قائمة المكونات", on_click=toggle_message_input, use_container_width=True)
st.button("🧠 رفع الصورة", on_click=toggle_message_upload, use_container_width=True)
st.button("📸 التقاط صورة", on_click=toggle_message_camera, use_container_width=True)
# إدخال المستخدم

 #عرض أو إخفاء الرسالة بناءً على حالة الجلسة
if st.session_state.show_message_input:
 ingredients_text = st.text_area("✍️ أدخل قائمة المكونات (يمكنك نسخها من الملصق):", height=200)
 st.session_state.show_message_camera = False
 st.session_state.show_message_upload = False

# 🟢 التقاط صورة بالكاميرا
 if st.session_state.show_message_upload:
  saved_image = get_ocr_from_camera()
  st.session_state.show_message_input = False
  st.session_state.show_message_camera= False

 
#رفع صورة لملصق المنتج
  if st.session_state.show_message_camera:
   saved_image = upload_image_ocr_from_folder()
   st.session_state.show_message_input = False
   st.session_state.show_message_upload = False

st.write("ارفع صورة لملصق المنتج وسنقوم بتحليل المكونات لمعرفة ما إذا كانت تحتوي على مشتقات من الحشرات.")

if saved_image:
 st.image(saved_image, caption="📷 الصورة التي تم رفعها", use_column_width=True)
else:
 st.warning("⚠️ لا توجد صورة محفوظة حتى الآن.")

if saved_image:
        ingredients_text = extract_text_from_image(saved_image)
 
  
# وضع الزر في العمود الأوسط

    # تحليل النص باستخدام GPT-4
st.button("🔍 تحليل النص", use_container_width=True)
         # if not ingredients_text.strip():
          # st.warning("يرجى إدخال مكونات أولاً")
with st.spinner("جاري التحليل باستخدام GPT-4..."):
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
