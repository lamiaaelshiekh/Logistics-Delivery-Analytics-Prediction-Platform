# # importing libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
# import pickle # تم استبداله بـ joblib لأنه أكثر كفاءة لمسارات Sci-kit Learn
import joblib # مكتبة joblib لتحميل النماذج المحفوظة
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO

# # إعدادات الصفحة
st.set_page_config(page_title="Logistics Analysis", layout='wide')

# ==============================================================================
# 🛠️ دالة تحميل البيانات والنماذج (Caching)
# ==============================================================================

# تحميل البيانات (يتم تخزينها مؤقتاً لمرة واحدة)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Delivery_Logistics_TimeFixed.csv')
        # التأكد من تحويل عمود التأخير إلى رقمي (1/0) إذا كان النص متوفراً
        if 'delayed' in df.columns:
            df['delayed_numeric'] = df['delayed'].apply(lambda x: 1 if x == 'yes' else 0)
        return df
    except FileNotFoundError:
        st.error("لم يتم العثور على ملف البيانات 'Delivery_Logistics_TimeFixed.csv'. يرجى التأكد من وضعه في نفس المجلد.")
        return pd.DataFrame()

# تحميل النماذج (يتم تخزينها مؤقتاً لمرة واحدة)
@st.cache_resource
def load_models():
    try:
        # تأكد من أن أسماء الملفات تتطابق مع الأسماء التي استخدمتها للحفظ
        reg_pipeline = joblib.load('reg_model_pipeline.joblib')
        class_pipeline = joblib.load('class_model_pipeline.joblib')
        return reg_pipeline, class_pipeline
    except FileNotFoundError:
        st.warning("⚠️ لم يتم العثور على ملفات النماذج (reg_model_pipeline.joblib أو class_model_pipeline.joblib). لن يعمل قسم التعلم الآلي حتى يتم وضع الملفات.")
        return None, None

df = load_data()
reg_pipeline, class_pipeline = load_models()

# # شريط جانبي
option = st.sidebar.selectbox("اختر القسم:", ['Home','Full Analysis','ML Prediction'])

# ==============================================================================
# -------------------------------- 🏠 HOME ------------------------------------
# ==============================================================================
if option == 'Home':
    st.title("📊 تطبيق تحليلات لوجستيات التوصيل")
    st.markdown("### 👨‍💻 Author: **Lamiaa Elshiekh**")
    st.write("هذه اللوحة المرئية تحلل وتتنبأ بحالة توصيل الطلبات: في الوقت المحدد أو متأخرة.")
    
    st.markdown("---")
    st.header("أول 5 صفوف من البيانات")
    if not df.empty:
        st.dataframe(df.head())
    
    st.markdown("---")
    st.header("ملخص إحصائي سريع")
    if not df.empty:
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        total_deliveries = len(df)
        avg_rating = df['delivery_rating'].mean().round(2)
        delay_rate = (df['delayed_numeric'].sum() / total_deliveries * 100).round(2) if 'delayed_numeric' in df.columns else 0

        col_kpi1.metric("إجمالي الشحنات", f"{total_deliveries}")
        col_kpi2.metric("متوسط التقييم", f"⭐ {avg_rating}")
        col_kpi3.metric("معدل التأخير العام", f"{delay_rate}%", delta=f"{delay_rate}% ارتفاع")

# ==============================================================================
# -------------------------------- 📈 Full Analysis -----------------------------
# ==============================================================================
elif option == 'Full Analysis':
    st.title("📈 التحليل الكامل للبيانات")
    
    if df.empty:
        st.warning("يرجى التأكد من تحميل ملف البيانات بنجاح لبدء التحليل.")
    else:
        # ===== FUNCTION 1: Advanced Analytics =====
        def advanced_analytics(df):
            st.subheader("🚀 التحليلات الذكية المتقدمة")

            # 1. تحليل الكفاءة الاقتصادية
            df['cost_per_km'] = df['delivery_cost'] / df['distance_km']
            df['cost_per_kg'] = df['delivery_cost'] / df['package_weight_kg']
            df['efficiency_score'] = (df['distance_km'] * df['package_weight_kg']) / df['delivery_cost']

            # 2. تصنيف الشركاء
            partner_stats = df.groupby('delivery_partner').agg({
                'delivery_rating': 'mean',
                'delivery_cost': 'mean',
                'delayed': lambda x: (x == 'yes').mean(),
                'delivery_id': 'count',
                'efficiency_score': 'mean'
            }).round(3)

            partner_stats.columns = [
                'avg_rating', 'avg_cost', 'delay_rate',
                'total_deliveries', 'efficiency'
            ]

            partner_stats['performance_tier'] = pd.cut(
                partner_stats['avg_rating'],
                bins=[0, 2, 3.5, 5],
                labels=['Low', 'Medium', 'High']
            )

            st.write("### 📊 ملخص أداء شركات التوصيل")
            st.dataframe(partner_stats)

            return df, partner_stats

        # ===== FUNCTION 2: Hidden Patterns =====
        def hidden_patterns(df):
            st.subheader("🔍 الأنماط المخفية")

            # 1. أفضل تركيب (منطقة + طقس + مركبة)
            df['combo'] = df['region'] + '_' + df['weather_condition'] + '_' + df['vehicle_type']
            
            combo_performance = df.groupby('combo').agg({
                'delivery_rating': 'mean',
                'delayed': lambda x: (x == 'yes').mean(),
                'delivery_id': 'count'
            }).sort_values('delivery_rating', ascending=False)

            st.write("### ⭐ أفضل 5 تركيبات (منطقة + طقس + مركبة)")
            st.dataframe(combo_performance.head())

            # 2. تحليل الحزم الحرجة
            weight_speed_corr = df['package_weight_kg'].corr(df['actual_delivery_hours'])
            st.write("### ⚖️ ارتباط وزن الحزمة بزمن التوصيل")
            st.write(f"قيمة الارتباط: **{weight_speed_corr:.3f}**")

            # 3. تأثير الطقس على الأداء
            weather_impact = df.groupby('weather_condition').agg({
                'delivery_rating': 'mean',
                'delayed': lambda x: (x == 'yes').mean(),
                'delivery_delay_hours': 'mean'
            })

            st.write("### 🌤 تأثير حالة الطقس على جودة التوصيل")
            st.dataframe(weather_impact)

            return combo_performance

        # ===== FUNCTION 3: Predictive Insights =====
        def predictive_insights(df):
            st.subheader("🎯 رؤى تنبؤية")

            # 1. خريطة حرارة التأخير
            delay_heatmap = df.pivot_table(
                index='delivery_mode',
                columns='weather_condition',
                values='delivery_delay_hours',
                aggfunc='mean'
            ).fillna(0)

            st.write("### 🔥 خريطة حرارة التأخير حسب (وسيلة التوصيل × حالة الطقس)")
            st.dataframe(delay_heatmap.round(2))

            # 2. أفضل الشركاء في الطقس السيء
            storm_performers = df[df['weather_condition'].isin(['stormy', 'rainy'])] \
                .groupby('delivery_partner').agg({
                    'delivery_rating': 'mean',
                    'delayed': lambda x: (x == 'yes').mean()
                }).sort_values('delivery_rating', ascending=False)

            st.write("### ⛈️ أفضل أداء للشركاء في ظروف الطقس السيء")
            st.dataframe(storm_performers.head())

            # 3. تحليل القيمة مقابل السعر
            df['value_score'] = (df['delivery_rating'] * 2) - (
                df['cost_per_km'] / df['cost_per_km'].max()
            )
            
            best_value = df.groupby('delivery_partner')['value_score'] \
                            .mean().sort_values(ascending=False)

            st.write("### 💰 أفضل قيمة مقابل السعر بين شركاء التوصيل")
            st.dataframe(best_value.head())

            return delay_heatmap, storm_performers, best_value

        # ===== FUNCTION 4: Partner Performance Plot =====
        def plot_partner_performance(df, partner_analytics):
            st.subheader("📈 أداء شركاء التوصيل")

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. التصنيف حسب التقييم
            partner_analytics['avg_rating'].sort_values().plot(
                kind='barh', ax=axes[0,0], color='skyblue'
            )
            axes[0,0].set_title('📊 Average Rating by Delivery Partner')
            axes[0,0].set_xlabel('Rating (out of 5)')
            
            # 2. معدل التأخير
            partner_analytics['delay_rate'].sort_values().plot(
                kind='barh', ax=axes[0,1], color='salmon'
            )
            axes[0,1].set_title('⏱️ Delay Rate by Partner')
            axes[0,1].set_xlabel('Delay Rate')
            
            # 3. تحليل التكلفة مقابل التقييم
            axes[1,0].scatter(
                partner_analytics['avg_cost'],
                partner_analytics['avg_rating'],
                s=partner_analytics['total_deliveries']*10,
                alpha=0.6
            )
            axes[1,0].set_title('💰 Cost vs Rating Analysis')
            axes[1,0].set_xlabel('Average Cost')
            axes[1,0].set_ylabel('Average Rating')
            
            # 4. مقياس الكفاءة
            partner_analytics['efficiency'].sort_values().plot(
                kind='barh', ax=axes[1,1], color='lightgreen'
            )
            axes[1,1].set_title('🚀 Partner Efficiency Score')
            axes[1,1].set_xlabel('Efficiency Score')
            
            plt.tight_layout()
            st.pyplot(fig)

        # ===== FUNCTION 5: Geographical Analysis =====
        def plot_geographical_analysis(df):
            st.subheader("🌍 Geographical Delivery Analysis")

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Performance by Region
            region_performance = df.groupby('region').agg({
                'delivery_rating': 'mean',
                'delayed': lambda x: (x == 'yes').mean(),
                'delivery_id': 'count'
            })
            
            axes[0,0].bar(region_performance.index, region_performance['delivery_rating'],
                            color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', 'violet'])
            axes[0,0].set_title('📍 Average Rating by Region')
            axes[0,0].set_ylabel('Average Rating')
            
            # 2. Delay Rate by Region
            axes[0,1].bar(region_performance.index, region_performance['delayed'],
                            color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', 'violet'])
            axes[0,1].set_title('⚠️ Delay Rate by Region')
            axes[0,1].set_ylabel('Delay Rate')
            
            # 3. Weather Impact
            weather_impact = df.groupby('weather_condition')['delivery_rating'].mean().sort_values()
            axes[1,0].bar(weather_impact.index, weather_impact.values,
                            color=['lightblue', 'lightgreen', 'lightcoral', 'gold', 'plum', 'gray'])
            axes[1,0].set_title('🌤️ Weather Impact on Delivery Rating')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # 4. Delivery Time Distribution
            axes[1,1].hist(df['actual_delivery_hours'], bins=30, alpha=0.7,
                            color='purple', edgecolor='black')
            axes[1,1].set_title('⏰ Delivery Time Distribution (Hours)')
            axes[1,1].set_xlabel('Delivery Time (Hours)')
            axes[1,1].set_ylabel('Number of Deliveries')

            plt.tight_layout()
            st.pyplot(fig)

        # ===== FUNCTION 6: Service Analysis =====
        def plot_service_analysis(df):
            st.subheader("🛠️ Service & Delivery Mode Analysis")

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Delivery Mode Performance
            delivery_mode_stats = df.groupby('delivery_mode').agg({
                'delivery_rating': 'mean',
                'delayed': lambda x: (x == 'yes').mean(),
                'delivery_cost': 'mean'
            })
            
            x = np.arange(len(delivery_mode_stats))
            width = 0.25
            
            axes[0,0].bar(x, delivery_mode_stats['delivery_rating'], width, label='Rating', alpha=0.8)
            axes[0,0].bar(x + width, delivery_mode_stats['delayed'], width, label='Delay Rate', alpha=0.8)
            axes[0,0].set_title('🚚 Delivery Mode Performance Comparison')
            axes[0,0].set_xticks(x + width / 2)
            axes[0,0].set_xticklabels(delivery_mode_stats.index)
            axes[0,0].legend()
            
            # 2. Cost by Delivery Mode
            axes[0,1].bar(delivery_mode_stats.index, delivery_mode_stats['delivery_cost'],
                            color=['red', 'blue', 'green', 'orange'])
            axes[0,1].set_title('💰 Average Cost by Delivery Mode')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # 3. Package Weight Distribution
            axes[1,0].hist(df['package_weight_kg'], bins=30, alpha=0.7,
                            color='teal', edgecolor='black')
            axes[1,0].set_title('⚖️ Package Weight Distribution')
            axes[1,0].set_xlabel('Weight (kg)')
            axes[1,0].set_ylabel('Frequency')
            
            # 4. Weight vs Delivery Time
            axes[1,1].scatter(df['package_weight_kg'], df['actual_delivery_hours'],
                            alpha=0.5, color='brown')
            axes[1,1].set_title('📦 Weight vs Delivery Time Relationship')
            axes[1,1].set_xlabel('Weight (kg)')
            axes[1,1].set_ylabel('Delivery Time (Hours)')
            
            plt.tight_layout()
            st.pyplot(fig)

        # ===== FUNCTION 7: Heatmaps =====
        def plot_heatmaps(df):
            st.subheader("🌡️ Heatmap Analysis: Delay & Rating Patterns")

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # 1. Delay Heatmap
            delay_pivot = df.pivot_table(
                index='region', 
                columns='weather_condition', 
                values='delivery_delay_hours', 
                aggfunc='mean'
            ).fillna(0)
            
            sns.heatmap(delay_pivot, annot=True, cmap='Reds', ax=axes[0], fmt='.2f')
            axes[0].set_title('🔥 Average Delay (Hours) - Region vs Weather')
            
            # 2. Rating Heatmap
            rating_pivot = df.pivot_table(
                index='delivery_mode', 
                columns='vehicle_type', 
                values='delivery_rating', 
                aggfunc='mean'
            ).fillna(0)
            
            sns.heatmap(rating_pivot, annot=True, cmap='Greens', ax=axes[1], fmt='.2f')
            axes[1].set_title('⭐ Average Rating - Service vs Vehicle Type')
            
            plt.tight_layout()
            st.pyplot(fig)

        # ===== FUNCTION 8: Dashboard Summary =====
        def plot_dashboard_summary(df):
            st.subheader("📊 Overall Dashboard Summary")

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            # 1. Rating Distribution
            df['delivery_rating'].value_counts().sort_index().plot(kind='bar', ax=axes[0,0], color='gold')
            axes[0,0].set_title('⭐ Rating Distribution')
            axes[0,0].set_xlabel('Rating')
            axes[0,0].set_ylabel('Number of Deliveries')
            
            # 2. Delivery Status (باستخدام delayed_numeric)
            if 'delayed_numeric' in df.columns:
                 status_counts = df['delayed_numeric'].map({1: 'Delayed', 0: 'On Time'}).value_counts()
                 status_counts.plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%', colors=['#66c2a5', '#fc8d62'])
            else: # Fallback if status column is missing
                 pd.Series({'Unknown': len(df)}).plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%')

            axes[0,1].set_title('📦 Delivery Status')
            axes[0,1].set_ylabel("")
            
            # 3. Vehicle Type Distribution
            df['vehicle_type'].value_counts().plot(kind='bar', ax=axes[0,2], color='lightblue')
            axes[0,2].set_title('🚗 Vehicle Type Distribution')
            axes[0,2].tick_params(axis='x', rotation=45)
            
            # 4. Cost vs Distance
            axes[1,0].scatter(df['distance_km'], df['delivery_cost'], alpha=0.6, color='purple')
            axes[1,0].set_title('💰 Cost vs Distance Analysis')
            axes[1,0].set_xlabel('Distance (km)')
            axes[1,0].set_ylabel('Cost')
            
            # 5. Delay by Service Type
            df.groupby('delivery_mode')['delivery_delay_hours'].mean().plot(kind='bar', ax=axes[1,1], color='orange')
            axes[1,1].set_title('⏱️ Average Delay by Service Type')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            # 6. Delivery Time Density
            axes[1,2].hist(df['actual_delivery_hours'], bins=30, alpha=0.7, color='green', density=True)
            axes[1,2].set_title('📈 Delivery Time Density')
            axes[1,2].set_xlabel('Delivery Time (Hours)')
            
            plt.tight_layout()
            st.pyplot(fig)

        # ===== استدعاء كل الـ Functions هنا بس! =====
        df_copy = df.copy()
        try: # محاولة حساب الميزات المضافة في التحليل الكامل
            df_copy['cost_per_km'] = df_copy['delivery_cost'] / df_copy['distance_km']
            df_copy['cost_per_kg'] = df_copy['delivery_cost'] / df_copy['package_weight_kg']
        except Exception:
            pass # قد تفشل إذا كانت الأعمدة العددية غير موجودة أو صفراً

        df_copy, partner_analytics = advanced_analytics(df_copy) 
        patterns = hidden_patterns(df_copy)
        delay_heat, storm_perf, value_scores = predictive_insights(df_copy)
        plot_partner_performance(df_copy, partner_analytics)
        plot_geographical_analysis(df_copy)
        plot_service_analysis(df_copy)
        plot_heatmaps(df_copy)
        plot_dashboard_summary(df_copy)


# ==============================================================================
# -------------------------------- 🤖 ML Prediction ------------------------------
# ==============================================================================
elif option == 'ML Prediction':
    st.title("🤖 التنبؤ بحالة الشحنة (Regression & Classification)")
    st.markdown("هذا القسم يستخدم نماذج التعلم الآلي للتنبؤ بساعات التأخير (Regression) واحتمالية التأخير (Classification) لشحنة جديدة.")
    
    if reg_pipeline is None or class_pipeline is None:
        st.error("🚨 لا يمكن إجراء التنبؤ. يرجى مراجعة التحذير في الأعلى والتأكد من وجود ملفات النماذج المحفوظة.")
    else:
        st.header("إدخال تفاصيل الشحنة")
        
        # قائمة الميزات (يجب أن تتطابق مع الميزات المستخدمة في التدريب)
        PARTNERS = ['delhivery', 'xpressbees', 'shadowfax', 'dhl', 'ecom express', 'fedex']
        PKG_TYPES = ['automobile parts', 'cosmetics', 'groceries', 'electronics', 'clothing', 'books', 'documents', 'heavy machinery']
        VEHICLES = ['bike', 'ev van', 'truck', 'van', 'car', 'drone']
        MODES = ['same day', 'express', 'two day', 'next day']
        REGIONS = ['west', 'central', 'east', 'north', 'south']
        WEATHERS = ['clear', 'cold', 'rainy', 'foggy', 'snowy', 'stormy']

        col1, col2, col3 = st.columns(3)

        # المدخلات الفئوية
        with col1:
            partner = st.selectbox("شريك التوصيل", PARTNERS)
            pkg_type = st.selectbox("نوع الحزمة", PKG_TYPES)
            vehicle = st.selectbox("نوع المركبة", VEHICLES)

        with col2:
            delivery_mode = st.selectbox("وضع التوصيل", MODES)
            region = st.selectbox("المنطقة", REGIONS)
            weather = st.selectbox("حالة الطقس", WEATHERS)

        # المدخلات العددية
        with col3:
            distance = st.number_input("المسافة بالكيلومتر (Distance_km)", min_value=1.0, max_value=300.0, value=150.0)
            weight = st.number_input("وزن الحزمة بالكيلوغرام (Package Weight_kg)", min_value=0.1, max_value=50.0, value=5.0)
            # 🛑 هذا هو الحقل الذي كان مفقوداً وتسبب في الخطأ
            cost = st.number_input("تكلفة التوصيل ($) (Delivery Cost)", min_value=1.0, max_value=500.0, value=50.0)
            rating = st.slider("تقييم التوصيل (Delivery Rating)", min_value=1, max_value=5, value=4)
            expected_hours = st.number_input("ساعات التوصيل المتوقعة (Expected Delivery Hours)", min_value=1.0, max_value=48.0, value=8.0)


        # زر التنبؤ
        if st.button("🚀 إجراء التنبؤ", type="primary"):
            
            # 📥 تجهيز بيانات الإدخال
            # 🛑 تم إضافة 'delivery_cost': cost هنا
            input_data = pd.DataFrame([{
                'delivery_partner': partner,
                'package_type': pkg_type,
                'vehicle_type': vehicle,
                'delivery_mode': delivery_mode,
                'region': region,
                'weather_condition': weather,
                'distance_km': distance,
                'package_weight_kg': weight,
                'delivery_cost': cost,  # 👈 الإضافة الجديدة لحل المشكلة
                'delivery_rating': rating,
                'expected_delivery_hours': expected_hours
            }])
            
            try:
                st.subheader("نتائج التنبؤ 🔮")

                # 1. تنبؤ الـ Regression (ساعات التأخير)
                reg_prediction = reg_pipeline.predict(input_data)[0]

                # 2. تنبؤ الـ Classification (تأخير / لا تأخير)
                class_prediction_label = class_pipeline.predict(input_data)[0]
                # احتمال التنبؤ (احتمال وجود تأخير)
                class_prediction_proba = class_pipeline.predict_proba(input_data)[0]
                
                # 0 = لا تأخير / 1 = تأخير 
                class_status = "تأخير محتمل 🔴" if class_prediction_label == 1 else "لا تأخير متوقع 🟢"
                prob_delay = class_prediction_proba[1] * 100 # احتمال التأخير (القيمة عند الفهرس 1)
                
                # 📊 عرض النتائج
                
                col_res1, col_res2 = st.columns(2)

                with col_res1:
                    st.markdown("### تنبؤ التأخير بالساعات (Regression)")
                    # نضبط قيمة التأخير إذا كانت سالبة (وصول مبكر) أو موجبة (تأخير)
                    delta_text = f"{reg_prediction:.2f} ساعة"
                    if reg_prediction > 0.05:
                         delta_color = "inverse"
                         st.error("يتوقع النموذج تأخيراً يرجى المتابعة.")
                    elif reg_prediction < -0.05:
                         delta_color = "normal"
                         delta_text = f"{-reg_prediction:.2f} ساعة مبكراً"
                         st.success("من المتوقع وصول الشحنة مبكراً.")
                    else:
                         delta_color = "off"
                         delta_text = "في الموعد"
                         st.info("من المتوقع وصول الشحنة في الموعد.")


                    st.metric(
                        label="ساعات التأخير المتوقعة",
                        value=f"{reg_prediction:.2f} ساعة",
                        delta=delta_text,
                        delta_color=delta_color
                    )
                    
                    
                with col_res2:
                    st.markdown("### تنبؤ حالة التأخير (Classification)")
                    
                    color_status = "red" if class_prediction_label == 1 else "green"
                    
                    st.markdown(f"#### <span style='color: {color_status};'>{class_status}</span>", unsafe_allow_html=True)
                    
                    st.metric(
                        label="احتمال التأخير",
                        value=f"{prob_delay:.1f}%"
                    )
                    
                    if class_prediction_label == 1:
                        st.warning("تصنيف النموذج: **متأخر (Delayed)**. يلزم اتخاذ إجراء.")
                    else:
                        st.success("تصنيف النموذج: **في الوقت المحدد (On Time)**. التوصيل يسير كما هو مخطط.")

            except Exception as e:
                st.error(f"حدث خطأ أثناء إجراء التنبؤ. قد تكون هناك مشكلة في كيفية تحميل أو استخدام النموذج. الخطأ: {e}")