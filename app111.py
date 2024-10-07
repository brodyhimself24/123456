import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置页面宽度
st.set_page_config(layout="wide")

# Streamlit界面
def main():
    # 设置页面标题
    st.title('八极称人体健康指标大模型')

    # 设置页面描述
    st.markdown("""
        <style>
            .stApp {
                background-color: #F9F9F9;
                padding: 20px;
                font-family: Arial, sans-serif;
            }
            .title {
                color: #333333;
                text-align: center;
            }
            .subtitle {
                color: #666666;
                text-align: center;
            }
            .input-label {
                color: #333333;
            }
            .result-label {
                color: #333333;
                font-weight: bold;
            }
            .result-value {
                color: #007BFF;
            }
        </style>
        """, unsafe_allow_html=True)

    # 设置副标题
    st.markdown('<p class="subtitle">请输入您的个人信息和测量数据，我们将为您预测健康指标。</p>', unsafe_allow_html=True)

    # 输入框布局
    col1, col2 = st.columns([1, 1])

    with col1:
        # 创建文本输入框供用户输入特征数据
        weight = st.number_input("体重 (kg)", min_value=0.0, max_value=150.0, value=65.0, step=0.1, format="%.1f")
        height = st.number_input("身高 (cm)", min_value=0.0, max_value=250.0, value=175.0, step=0.1, format="%.1f")
        age = st.number_input("年龄", min_value=0, max_value=120, value=25)
        ra5 = st.number_input("5Hz RA值", min_value=0.0, max_value=500.0, value=100.0, step=0.1, format="%.1f")
        la5 = st.number_input("5Hz LA值", min_value=0.0, max_value=500.0, value=100.0, step=0.1, format="%.1f")
        tr5 = st.number_input("5Hz TR值", min_value=0.0, max_value=500.0, value=50.0, step=0.1, format="%.1f")
        rl5 = st.number_input("5Hz RL值", min_value=0.0, max_value=500.0, value=120.0, step=0.1, format="%.1f")
        ll5 = st.number_input("5Hz LL值", min_value=0.0, max_value=500.0, value=120.0, step=0.1, format="%.1f")

    with col2:
        ra50 = st.number_input("50Hz RA值", min_value=0.0, max_value=500.0, value=100.0, step=0.1, format="%.1f")
        la50 = st.number_input("50Hz LA值", min_value=0.0, max_value=500.0, value=100.0, step=0.1, format="%.1f")
        tr50 = st.number_input("50Hz TR值", min_value=0.0, max_value=500.0, value=50.0, step=0.1, format="%.1f")
        rl50 = st.number_input("50Hz RL值", min_value=0.0, max_value=500.0, value=120.0, step=0.1, format="%.1f")
        ll50 = st.number_input("50Hz LL值", min_value=0.0, max_value=500.0, value=120.0, step=0.1, format="%.1f")
        ra250 = st.number_input("250Hz RA值", min_value=0.0, max_value=500.0, value=100.0, step=0.1, format="%.1f")
        la250 = st.number_input("250Hz LA值", min_value=0.0, max_value=500.0, value=100.0, step=0.1, format="%.1f")
        tr250 = st.number_input("250Hz TR值", min_value=0.0, max_value=500.0, value=50.0, step=0.1, format="%.1f")
        rl250 = st.number_input("250Hz RL值", min_value=0.0, max_value=500.0, value=120.0, step=0.1, format="%.1f")
        ll250 = st.number_input("250Hz LL值", min_value=0.0, max_value=500.0, value=120.0, step=0.1, format="%.1f")

    # 用户点击预测按钮
    if st.button('进行预测'):
        # 加载模型和标准化器
        try:
            model = load('heu12.joblib')  # 确保这是正确的文件名
            scaler_X = load('scaler_X12.joblib')  # 确保这是正确的文件名
        except FileNotFoundError:
            st.error("模型或标准化器文件未找到，请检查文件路径。")
            return
        
        # 构建特征数组，并确保是二维形状
        features = np.array([
            weight, height, age, ra5, la5, tr5, rl5, ll5, ra50, la50, tr50, rl50, ll50, ra250, la250, tr250, rl250, ll250
        ]).reshape(1, -1)

        # 输出特征的形状以进行验证
        # st.write(f"输入特征的形状: {features.shape}")

        # 使用加载的 StandardScaler 实例进行特征标准化
        features_scaled = scaler_X.transform(features)

        # 输出标准化后的特征的形状以进行验证
        # st.write(f"标准化后的特征的形状: {features_scaled.shape}")

        # 使用模型进行预测
        try:
            prediction = model.predict(features_scaled)
        except ValueError as e:
            st.error(f"预测时发生错误：{str(e)}")
            return

        # 计算水分含量
        water_percentage = prediction[0, 3]  # 获取水分率百分比
        water_content = weight * (water_percentage / 100)  # 计算水分含量
        
        # 计算 BMI
        bmi = weight / ((height / 100) ** 2)  # 身高单位为 cm，转换为 m
        
        # 添加 "水分含量" 和 "BMI" 标签
        labels_with_units = [
            '体脂率', '肌肉', '骨骼肌', '水分率百分比', '蛋白质', '无机盐', '内脏脂肪', 
            '基础代谢', '节段肌肉左臂', '节段肌肉右臂', '节段肌肉躯干', '节段肌肉左腿', '节段肌肉右腿', '骨盐量', '腰臀比',
            '水分含量', 'BMI'
        ]

        # 将水分含量和 BMI 添加到预测结果中
        predictions_with_water_and_bmi = np.hstack((prediction[0], [water_content, bmi]))

        # 定义每个标签的单位
        units = [
            '%', 'kg', 'kg', '%', 'kg', 'kg', '',  # 内脏脂肪没有单位
            '', 'kg', 'kg', 'kg', 'kg', 'kg', 'kg', ''      # 腰臀比没有单位
        ]

        # 结果布局
        col_result1, col_result2, col_result3 = st.columns([1, 1, 1])

        # 显示预测结果
        st.markdown('<p class="result-label">预测结果如下：</p>', unsafe_allow_html=True)
        for i, (label, unit) in enumerate(zip(labels_with_units, units + ['kg', ''])):
            # 根据标签添加单位
            if unit:
                col_index = i % 3
                if col_index == 0:
                    col_result1.write(f"{label}: {predictions_with_water_and_bmi[i]:.2f} {unit}")
                elif col_index == 1:
                    col_result2.write(f"{label}: {predictions_with_water_and_bmi[i]:.2f} {unit}")
                else:
                    col_result3.write(f"{label}: {predictions_with_water_and_bmi[i]:.2f} {unit}")
            else:
                col_index = i % 3
                if col_index == 0:
                    col_result1.write(f"{label}: {predictions_with_water_and_bmi[i]:.2f}")
                elif col_index == 1:
                    col_result2.write(f"{label}: {predictions_with_water_and_bmi[i]:.2f}")
                else:
                    col_result3.write(f"{label}: {predictions_with_water_and_bmi[i]:.2f}")

if __name__ == "__main__":
    main()