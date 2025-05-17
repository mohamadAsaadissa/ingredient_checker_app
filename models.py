from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

# تعريف القاعدة الأساسية
Base = declarative_base()

# تعريف نموذج الجدول
class OCRImage(Base):
    __tablename__ = 'ocr_extracted_text'
    id = Column(Integer, primary_key=True)
    extracted_text = Column(String, nullable=False)

# دالة لإنشاء قاعدة البيانات والجداول
def create_new_dbsqlite(db_path='sqlite:///ocr_images.db'):
    # إنشاء محرك قاعدة البيانات
    engine = create_engine(db_path, echo=True)

    # إنشاء الجداول إذا لم تكن موجودة مسبقًا
    Base.metadata.create_all(engine, checkfirst=True)

    # إنشاء جلسة للتعامل مع قاعدة البيانات
    Session = sessionmaker(bind=engine)
    session = Session()

    return session
