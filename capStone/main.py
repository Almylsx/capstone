from pycaret.datasets import get_data
from pycaret.classification import setup, compare_models
import streamlit as st

# جلب بيانات المثال
data = get_data('diabetes')

# إعداد PyCaret
model_setup = setup(data=data, target='Class variable', silent=True, session_id=123)

# تدريب النموذج واختيار الأفضل
best_model = compare_models()




# تعريف وظيفة لتحميل وتدريب النموذج
def train_model():
    data = get_data('diabetes')
    model_setup = setup(data=data, target='Class variable', silent=True, session_id=123)
    best_model = compare_models()
    return best_model

# إنشاء عناوين وأزرار في واجهة Streamlit
st.title('تطبيق تعلم الآلة باستخدام PyCaret')
if st.button('تدريب النموذج'):
    with st.spinner('جاري تدريب النموذج...'):
        model = train_model()
        st.success('تم تدريب النموذج بنجاح!')
        st.write(model)
