import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np

# تحميل البيانات
@st.cache_data
def load_data():
    # قراءة ملف Excel
    df = pd.read_excel('TestTestMarketing.xlsx', sheet_name='Sheet1')
    
    # تنظيف البيانات
    df = df.dropna(subset=['Date'])  # حذف الصفوف التي تحتوي على تاريخ فارغ
    df['Date'] = pd.to_datetime(df['Date'])  # تحويل العمود إلى نوع تاريخ
    
    # استخراج معلومات إضافية من التاريخ
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Day_Name'] = df['Date'].dt.day_name()
    df['Week'] = df['Date'].dt.isocalendar().week
    
    return df

# تحميل البيانات
df = load_data()

# واجهة التطبيق
st.title('📊 تحليلات التسويق العقاري')
st.markdown("""
هذا التطبيق يساعدك في تحليل بيانات الليدرز التسويقية وتتبع أداء الفريق.
""")

# شريط جانبي للتصفية
st.sidebar.header('خيارات التصفية')

# فلترة حسب التاريخ
date_range = st.sidebar.date_input(
    "اختر نطاق تاريخي",
    value=[df['Date'].min().date(), df['Date'].max().date()],
    min_value=df['Date'].min().date(),
    max_value=df['Date'].max().date()
)

# تحويل التواريخ المحددة إلى datetime
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

# فلترة حسب المسوق مع خيار "الكل"
marketer_default = []  # افتراضيًا لا يتم اختيار أي مسوق
marketers = st.sidebar.multiselect(
    'اختر المسوقين (اترك فارغًا لاختيار الكل)',
    options=df['Marketer'].unique(),
    default=marketer_default
)

# فلترة حسب المنطقة مع خيار "الكل"
location_default = []  # افتراضيًا لا يتم اختيار أي منطقة
locations = st.sidebar.multiselect(
    'اختر المناطق (اترك فارغًا لاختيار الكل)',
    options=df['Location'].unique(),
    default=location_default
)

# فلترة حسب المشروع مع خيار "الكل"
project_default = []  # افتراضيًا لا يتم اختيار أي مشروع
projects = st.sidebar.multiselect(
    'اختر المشاريع (اترك فارغًا لاختيار الكل)',
    options=df['Project'].unique(),
    default=project_default
)

# فلترة حسب السيلز مع خيار "الكل"
sales_default = []  # افتراضيًا لا يتم اختيار أي سيلز
sales = st.sidebar.multiselect(
    'اختر السيلز (اترك فارغًا لاختيار الكل)',
    options=df['Sales'].unique(),
    default=sales_default
)

# تطبيق الفلاتر
filtered_df = df[
    (df['Date'] >= start_date) & 
    (df['Date'] <= end_date)
]

# تطبيق الفلاتر الاختيارية
if marketers:
    filtered_df = filtered_df[filtered_df['Marketer'].isin(marketers)]
if locations:
    filtered_df = filtered_df[filtered_df['Location'].isin(locations)]
if projects:
    filtered_df = filtered_df[filtered_df['Project'].isin(projects)]
if sales:
    filtered_df = filtered_df[filtered_df['Sales'].isin(sales)]

# عرض البيانات المصفاة
st.subheader('البيانات المصفاة')
st.dataframe(filtered_df)

# تحميل البيانات
st.download_button(
    label="تحميل البيانات المصفاة",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name='filtered_marketing_data.csv',
    mime='text/csv'
)

# إحصائيات أساسية
st.subheader('📈 إحصائيات أساسية')

col1, col2, col3 = st.columns(3)
col1.metric("إجمالي المكالمات", filtered_df.shape[0])
col2.metric("عدد المسوقين", filtered_df['Marketer'].nunique())
col3.metric("عدد المشاريع", filtered_df['Project'].nunique())

col4, col5, col6 = st.columns(3)
col4.metric("عدد المناطق", filtered_df['Location'].nunique())
col5.metric("عدد السيلز", filtered_df['Sales'].nunique())
col6.metric("الفترة الزمنية", f"{start_date.date()} إلى {end_date.date()}")

# تحليل المكالمات اليومية
st.subheader('📞 توزيع المكالمات اليومية')
daily_calls = filtered_df.groupby('Date').size().reset_index(name='Count')

fig_daily = px.line(
    daily_calls, 
    x='Date', 
    y='Count',
    title='عدد المكالمات حسب اليوم',
    labels={'Date': 'التاريخ', 'Count': 'عدد المكالمات'}
)
st.plotly_chart(fig_daily, use_container_width=True)

# تحليل المكالمات حسب المسوق
st.subheader('👥 توزيع المكالمات حسب المسوق')
marketer_counts = filtered_df['Marketer'].value_counts().reset_index()
marketer_counts.columns = ['Marketer', 'Count']

fig_marketer = px.bar(
    marketer_counts,
    x='Marketer',
    y='Count',
    title='عدد المكالمات لكل مسوق',
    labels={'Marketer': 'المسوق', 'Count': 'عدد المكالمات'},
    color='Marketer'
)
st.plotly_chart(fig_marketer, use_container_width=True)

# تحليل المكالمات حسب المنطقة
st.subheader('📍 توزيع المكالمات حسب المنطقة')
location_counts = filtered_df['Location'].value_counts().reset_index()
location_counts.columns = ['Location', 'Count']

fig_location = px.pie(
    location_counts,
    names='Location',
    values='Count',
    title='نسبة المكالمات حسب المنطقة'
)
st.plotly_chart(fig_location, use_container_width=True)

# تحليل المكالمات حسب المشروع
st.subheader('🏢 توزيع المكالمات حسب المشروع')
project_counts = filtered_df['Project'].value_counts().reset_index()
project_counts.columns = ['Project', 'Count']

# عرض أهم 10 مشاريع فقط لتجنب ازدحام الرسم البياني
top_projects = project_counts.head(10)

fig_project = px.bar(
    top_projects,
    x='Project',
    y='Count',
    title='أهم 10 مشاريع حسب عدد المكالمات',
    labels={'Project': 'المشروع', 'Count': 'عدد المكالمات'},
    color='Project'
)
st.plotly_chart(fig_project, use_container_width=True)

# تحليل المكالمات حسب السيلز
st.subheader('👔 توزيع المكالمات حسب السيلز')
sales_counts = filtered_df['Sales'].value_counts().reset_index()
sales_counts.columns = ['Sales', 'Count']

# عرض أهم 10 سيلز فقط
top_sales = sales_counts.head(10)

fig_sales = px.bar(
    top_sales,
    x='Sales',
    y='Count',
    title='أهم 10 سيلز حسب عدد المكالمات',
    labels={'Sales': 'السيلز', 'Count': 'عدد المكالمات'},
    color='Sales'
)
st.plotly_chart(fig_sales, use_container_width=True)

# تحليل المكالمات حسب الشهر
st.subheader('🗓️ توزيع المكالمات الشهري')
monthly_calls = filtered_df.groupby(['Year', 'Month']).size().reset_index(name='Count')
monthly_calls['Month_Name'] = monthly_calls['Month'].apply(lambda x: datetime(2023, x, 1).strftime('%B'))

fig_monthly = px.bar(
    monthly_calls,
    x='Month_Name',
    y='Count',
    title='عدد المكالمات حسب الشهر',
    labels={'Month_Name': 'الشهر', 'Count': 'عدد المكالمات'},
    color='Month_Name'
)
st.plotly_chart(fig_monthly, use_container_width=True)

# تحليل المكالمات حسب اليوم في الأسبوع
st.subheader('📅 توزيع المكالمات حسب اليوم في الأسبوع')
day_counts = filtered_df['Day_Name'].value_counts().reset_index()
day_counts.columns = ['Day_Name', 'Count']

fig_day = px.bar(
    day_counts,
    x='Day_Name',
    y='Count',
    title='عدد المكالمات حسب اليوم في الأسبوع',
    labels={'Day_Name': 'اليوم', 'Count': 'عدد المكالمات'},
    color='Day_Name'
)
st.plotly_chart(fig_day, use_container_width=True)

# تحليل العلاقة بين المسوق والمنطقة
st.subheader('🔗 العلاقة بين المسوق والمنطقة')
marketer_location = filtered_df.groupby(['Marketer', 'Location']).size().reset_index(name='Count')

fig_marketer_location = px.sunburst(
    marketer_location,
    path=['Marketer', 'Location'],
    values='Count',
    title='توزيع المكالمات بين المسوقين والمناطق'
)
st.plotly_chart(fig_marketer_location, use_container_width=True)

# تحليل العلاقة بين المسوق والمشروع
st.subheader('🏗️ العلاقة بين المسوق والمشروع')
marketer_project = filtered_df.groupby(['Marketer', 'Project']).size().reset_index(name='Count')

# عرض أهم 20 مجموعة فقط
marketer_project = marketer_project.sort_values('Count', ascending=False).head(20)

fig_marketer_project = px.bar(
    marketer_project,
    x='Marketer',
    y='Count',
    color='Project',
    title='توزيع المكالمات بين المسوقين والمشاريع',
    labels={'Marketer': 'المسوق', 'Count': 'عدد المكالمات', 'Project': 'المشروع'}
)
st.plotly_chart(fig_marketer_project, use_container_width=True)

# تحليل الاتجاهات الزمنية للمسوقين
st.subheader('📊 اتجاهات أداء المسوقين مع الوقت')
marketer_trend = filtered_df.groupby(['Date', 'Marketer']).size().reset_index(name='Count')

fig_marketer_trend = px.line(
    marketer_trend,
    x='Date',
    y='Count',
    color='Marketer',
    title='اتجاهات أداء المسوقين مع الوقت',
    labels={'Date': 'التاريخ', 'Count': 'عدد المكالمات', 'Marketer': 'المسوق'}
)
st.plotly_chart(fig_marketer_trend, use_container_width=True)

# تحليل أداء المسوقين حسب المشروع
st.subheader('📌 أداء المسوقين حسب المشروع')
selected_marketer = st.selectbox(
    'اختر مسوقًا لعرض أدائه حسب المشروع',
    options=filtered_df['Marketer'].unique()
)

marketer_project_perf = filtered_df[filtered_df['Marketer'] == selected_marketer]
marketer_project_perf = marketer_project_perf.groupby('Project').size().reset_index(name='Count')

fig_marketer_project_perf = px.pie(
    marketer_project_perf,
    names='Project',
    values='Count',
    title=f'توزيع مكالمات {selected_marketer} حسب المشروع'
)
st.plotly_chart(fig_marketer_project_perf, use_container_width=True)

# تحليل أداء المسوقين حسب المنطقة
st.subheader('🌍 أداء المسوقين حسب المنطقة')
marketer_location_perf = filtered_df[filtered_df['Marketer'] == selected_marketer]
marketer_location_perf = marketer_location_perf.groupby('Location').size().reset_index(name='Count')

fig_marketer_location_perf = px.bar(
    marketer_location_perf,
    x='Location',
    y='Count',
    title=f'توزيع مكالمات {selected_marketer} حسب المنطقة',
    labels={'Location': 'المنطقة', 'Count': 'عدد المكالمات'},
    color='Location'
)
st.plotly_chart(fig_marketer_location_perf, use_container_width=True)

# تحليل أداء المسوقين حسب السيلز
st.subheader('👔 أداء المسوقين حسب السيلز')
marketer_sales_perf = filtered_df[filtered_df['Marketer'] == selected_marketer]
marketer_sales_perf = marketer_sales_perf.groupby('Sales').size().reset_index(name='Count')

fig_marketer_sales_perf = px.bar(
    marketer_sales_perf,
    x='Sales',
    y='Count',
    title=f'توزيع مكالمات {selected_marketer} حسب السيلز',
    labels={'Sales': 'السيلز', 'Count': 'عدد المكالمات'},
    color='Sales'
)
st.plotly_chart(fig_marketer_sales_perf, use_container_width=True)

# ------------------------------------------------------
# التحليلات الجديدة المطلوبة
# ------------------------------------------------------

# تحليل توزيع السيلز لمسوق معين
st.subheader('📊 توزيع السيلز لمسوق معين')

selected_marketer_for_sales = st.selectbox(
    'اختر مسوقًا لعرض توزيع السيلز له',
    options=filtered_df['Marketer'].unique(),
    key='marketer_sales_distribution'
)

# فلترة البيانات للمسوق المحدد
marketer_sales_dist = filtered_df[filtered_df['Marketer'] == selected_marketer_for_sales]

# حساب النسب المئوية
sales_dist = marketer_sales_dist['Sales'].value_counts(normalize=True).reset_index()
sales_dist.columns = ['Sales', 'Percentage']
sales_dist['Percentage'] = sales_dist['Percentage'] * 100

# رسم بياني دائري للنسب المئوية
fig_sales_dist = px.pie(
    sales_dist,
    names='Sales',
    values='Percentage',
    title=f'توزيع السيلز للمسوق {selected_marketer_for_sales} (النسب المئوية)',
    labels={'Sales': 'السيلز', 'Percentage': 'النسبة المئوية'}
)
st.plotly_chart(fig_sales_dist, use_container_width=True)

# رسم بياني شريطي للتوزيع
fig_sales_dist_bar = px.bar(
    sales_dist.sort_values('Percentage', ascending=False),
    x='Sales',
    y='Percentage',
    title=f'توزيع السيلز للمسوق {selected_marketer_for_sales}',
    labels={'Sales': 'السيلز', 'Percentage': 'النسبة المئوية'},
    color='Sales'
)
st.plotly_chart(fig_sales_dist_bar, use_container_width=True)

# تحليل توزيع المشاريع لمسوق معين
st.subheader('🏗️ توزيع المشاريع لمسوق معين')

selected_marketer_for_projects = st.selectbox(
    'اختر مسوقًا لعرض توزيع المشاريع له',
    options=filtered_df['Marketer'].unique(),
    key='marketer_projects_distribution'
)

# فلترة البيانات للمسوق المحدد
marketer_projects_dist = filtered_df[filtered_df['Marketer'] == selected_marketer_for_projects]

# حساب النسب المئوية للمشاريع (أعلى 10 فقط لتجنب الازدحام)
projects_dist = marketer_projects_dist['Project'].value_counts(normalize=True).reset_index()
projects_dist.columns = ['Project', 'Percentage']
projects_dist['Percentage'] = projects_dist['Percentage'] * 100
top_projects_dist = projects_dist.head(10)

# رسم بياني دائري للنسب المئوية
fig_projects_dist = px.pie(
    top_projects_dist,
    names='Project',
    values='Percentage',
    title=f'توزيع المشاريع للمسوق {selected_marketer_for_projects} (أهم 10 مشاريع)',
    labels={'Project': 'المشروع', 'Percentage': 'النسبة المئوية'}
)
st.plotly_chart(fig_projects_dist, use_container_width=True)

# رسم بياني شريطي للتوزيع
fig_projects_dist_bar = px.bar(
    top_projects_dist.sort_values('Percentage', ascending=False),
    x='Project',
    y='Percentage',
    title=f'توزيع المشاريع للمسوق {selected_marketer_for_projects} (أهم 10 مشاريع)',
    labels={'Project': 'المشروع', 'Percentage': 'النسبة المئوية'},
    color='Project'
)
st.plotly_chart(fig_projects_dist_bar, use_container_width=True)

# تحليل تفصيلي لأداء المسوق مع السيلز والمشاريع
st.subheader('📌 تحليل تفصيلي لأداء المسوق')

selected_marketer_detailed = st.selectbox(
    'اختر مسوقًا لعرض تحليل تفصيلي لأدائه',
    options=filtered_df['Marketer'].unique(),
    key='marketer_detailed_analysis'
)

# فلترة البيانات للمسوق المحدد
marketer_detailed = filtered_df[filtered_df['Marketer'] == selected_marketer_detailed]

# تحليل العلاقة بين السيلز والمشاريع للمسوق المحدد
sales_projects_dist = marketer_detailed.groupby(['Sales', 'Project']).size().reset_index(name='Count')

# عرض أهم 20 مجموعة فقط
top_sales_projects = sales_projects_dist.sort_values('Count', ascending=False).head(20)

fig_sales_projects = px.bar(
    top_sales_projects,
    x='Sales',
    y='Count',
    color='Project',
    title=f'توزيع المكالمات للسيلز والمشاريع للمسوق {selected_marketer_detailed}',
    labels={'Sales': 'السيلز', 'Count': 'عدد المكالمات', 'Project': 'المشروع'},
    barmode='stack'
)
st.plotly_chart(fig_sales_projects, use_container_width=True)

# جدول تفصيلي لأداء المسوق
st.subheader('📋 جدول تفصيلي لأداء المسوق')
marketer_summary = marketer_detailed.groupby(['Sales', 'Project', 'Location']).size().reset_index(name='Count')
st.dataframe(marketer_summary.sort_values('Count', ascending=False))