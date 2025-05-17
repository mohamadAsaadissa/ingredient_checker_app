# app.py

import easyocr
import streamlit as st
import openai
import pytesseract
from PIL import Image,ImageDraw
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import declarative_base, sessionmaker
import os
import uuid
import io
from models import OCRImage, create_new_dbsqlite###
from arabic_support import support_arabic_text
#pip install nest_asyncio
#للسماح بتشغيل الحلقات غير المتزامنة
#import nest_asyncio
#nest_asyncio.apply()

#هذا التعديل يساعد على تجاوز الخطأ المتعلق بـ torch.classes.

import torch
torch.classes.__path__ = []


# وظيفة لحساب تشابه النصوص
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
    return similarity[0][0]
# 🟢 التقاط صورة بالكاميرا
def get_ocr_from_camera():
    img_file = st.camera_input("التقط صورة")

    if img_file is not None:
        img = Image.open(img_file).convert("RGB")
        img_np = np.array(img)

        reader = easyocr.Reader(['ar', 'en'])
        with st.spinner("🔍 جارٍ تحليل الصورة..."):
            results = reader.readtext(img_np)

        draw = ImageDraw.Draw(img)
        for (bbox, text, confidence) in results:
            top_left = tuple(bbox[0])
            bottom_right = tuple(bbox[2])
            draw.rectangle([top_left, bottom_right], outline="red", width=3)

        st.image(img, caption="📄 الصورة مع المستطيلات حول النصوص", use_container_width=True)

        return img

    # st.warning("⚠️ لا توجد صورة محفوظة حتى الآن.")
    
#رفع صورة لملصق المنتج
def upload_image_ocr_from_folder():
    uploaded_file = st.file_uploader("📸: ارفع صورة ", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 الصورة التي تم رفعها", use_container_width=True)
            return image
        except Exception as e:
            st.error(f"حدث خطأ أثناء فتح الصورة: {e}")
            return None
    else:
        st.warning("لم يتم تحميل أي ملف.")
        return None
        
    #حويل الصورة إلى نص
def extract_text_from_image(saved_image):
    reader = easyocr.Reader(['ar', 'en'])
    img_np = np.array(saved_image)
    results = reader.readtext(img_np)

    extracted_text = "\n".join([text for (_, text, _) in results])

    st.subheader("📝 النصوص المكتشفة:")
    for (_, text, confidence) in results:
        st.write(f"- {text} (الدقة: {confidence:.2f})")

    st.text_area("📄 النص المستخرج من الصورة:", value=extracted_text, height=200)

    return extracted_text
    
#  تحليل المكونات باستخدام GPT-4
#def analyze_ingredients_with_gpt(ingredients_text):

    
    # دالة لتبديل حالة العرض
#def toggle_message_input():
 #   st.session_state.show_message_input = not st.session_state.show_message_input

def toggle_message_camera():
    st.session_state.show_message_camera = not st.session_state.show_message_camera

def toggle_message_upload():
    st.session_state.show_message_upload = not st.session_state.show_message_upload


# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY") or "ضع_مفتاحك_هنا"

# إذا كنت على Windows:
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# إنشاء الجداول في قاعدة البيانات (فقط إذا لم تكن موجودة)
session=create_new_dbsqlite('sqlite:///mydatabase.db')

st.set_page_config(page_title="تحليل المكونات الغذائية", page_icon="🐞", layout="centered")

# تفعيل دعم النصوص العربية في جميع المكونات
support_arabic_text(all=True)
# التحقق من وجود المفتاح في حالة الجلسة، وإذا لم يكن موجودًا، يتم تهيئته
#if 'show_message_input' not in st.session_state:
 #   st.session_state.show_message_input = True
if 'show_message_camera' not in st.session_state:
    st.session_state.show_message_camera = False
if 'show_message_upload' not in st.session_state:
    st.session_state.show_message_upload = True
# إنشاء ثلاثة أعمدة بنسبة عرض متساوية
col1, col2, col3 = st.columns([1, 4, 1])
# Default values


with col2:
 st.title("🐞 تحليل المكونات الغذائية")

st.write("تحقق مما إذا كانت قائمة المكونات تحتوي على مشتقات من الحشرات.")


    # زر لتبديل عرض الرسالة
#st.button("✍️ أدخل قائمة المكونات", on_click=toggle_message_input, use_container_width=True)
st.button("🧠 رفع الصورة", on_click=toggle_message_upload, use_container_width=True)
st.button("📸 التقاط صورة", on_click=toggle_message_camera, use_container_width=True)
# إدخال المستخدم

 #عرض أو إخفاء الرسالة بناءً على حالة الجلسة
#if st.session_state.show_message_input:
 #ingredients_text = st.text_area("✍️ أدخل قائمة المكونات (يمكنك نسخها من الملصق):", height=200)
 #st.session_state.show_message_camera = False
 #st.session_state.show_message_upload = False
#else:

# 🟢 التقاط صورة بالكاميرا
if st.session_state.show_message_upload:
    saved_image = upload_image_ocr_from_folder()
    st.session_state.show_message_camera = False

elif st.session_state.show_message_camera:
    saved_image = get_ocr_from_camera()
    st.session_state.show_message_upload = False

st.write("ارفع صورة لملصق المنتج وسنقوم بتحليل المكونات لمعرفة ما إذا كانت تحتوي على مشتقات من الحشرات.")



    # تحليل النص باستخدام GPT-4
if st.button("🔍 تحليل النص", use_container_width=True):
    if saved_image:
        with st.spinner("جاري استخراج النص..."):
            extracted_text = extract_text_from_image(saved_image)

        if not extracted_text.strip():
            st.warning("لم يتم العثور على نص قابل للاستخراج في الصورة.")
        else:
            # استرجاع جميع الصور من قاعدة البيانات
            images = session.query(OCRImage).all()
            similarities = []

            for img in images:
                similarity_score = calculate_similarity(extracted_text, img.extracted_text)
                similarities.append((img, similarity_score))

            # التحقق من وجود تطابق
            matched = False
            for img, score in similarities:
                if score >= 0.9:
                    matched = True
                    st.success(f"تم العثور على تطابق مع الصورة ذات المعرف {img.id} بدرجة تشابه {score:.2f}")
                    st.write("النص المستخرج مسبقًا:")
                    st.write(img.extracted_text)
                    break

          
                        st.success("✅ نتيجة التحليل:")
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"❌ خطأ أثناء الاتصال بـ GPT-4: {e}")
                    finally:
                        # حفظ النص المستخرج في قاعدة البيانات
                      #  new_entry = OCRImage(extracted_text=extracted_text)
                      #  session.add(new_entry)
                      #  session.commit()
                        st.success("لم يتم العثور على تطابق. تم حفظ النص المستخرج في قاعدة البيانات.")
    else:
        st.warning("⚠️ لا توجد صورة محفوظة حتى الآن.")
