import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Streamlit界面
def main():
    st.title('健康指标预测')

    # 创建文本输入框供用户输入特征数据
    weight = st.number_input("请输入体重 (kg)", min_value=0.0, max_value=150.0, value=65.0, step=0.1, format="%.1f")
    height = st.number_input("请输入身高 (cm)", min_value=0.0, max_value=250.0, value=175.0, step=0.1, format="%.1f")
    age = st.number_input("请输入年龄", min_value=0, max_value=120, value=25)
    ra5 = st.number_input("请输入5Hz的RA值", min_value=0.0, max_value=500.0, value=100.0, step=0.1, format="%.1f")
    la5 = st.number_input("请输入5Hz的LA值", min_value=0.0, max_value=500.0, value=100.0, step=0.1, format="%.1f")
    tr5 = st.number_input("请输入5Hz的TR值", min_value=0.0, max_value=500.0, value=50.0, step=0.1, format="%.1f")
    rl5 = st.number_input("请输入5Hz的RL值", min_value=0.0, max_value=500.0, value=120.0, step=0.1, format="%.1f")
    ll5 = st.number_input("请输入5Hz的LL值", min_value=0.0, max_value=500.0, value=120.0, step=0.1, format="%.1f")
    ra50 = st.number_input("请输入50Hz的RA值", min_value=0.0, max_value=500.0, value=100.0, step=0.1, format="%.1f")
    la50 = st.number_input("请输入50Hz的LA值", min_value=0.0, max_value=500.0, value=100.0, step=0.1, format="%.1f")
    tr50 = st.number_input("请输入50Hz的TR值", min_value=0.0, max_value=500.0, value=50.0, step=0.1, format="%.1f")
    rl50 = st.number_input("请输入50Hz的RL值", min_value=0.0, max_value=500.0, value=120.0, step=0.1, format="%.1f")
    ll50 = st.number_input("请输入50Hz的LL值", min_value=0.0, max_value=500.0, value=120.0, step=0.1, format="%.1f")
    ra250 = st.number_input("请输入250Hz的RA值", min_value=0.0, max_value=500.0, value=100.0, step=0.1, format="%.1f")
    la250 = st.number_input("请输入250Hz的LA值", min_value=0.0, max_value=500.0, value=100.0, step=0.1, format="%.1f")
    tr250 = st.number_input("请输入250Hz的TR值", min_value=0.0, max_value=500.0, value=50.0, step=0.1, format="%.1f")
    rl250 = st.number_input("请输入250Hz的RL值", min_value=0.0, max_value=500.0, value=120.0, step=0.1, format="%.1f")
    ll250 = st.number_input("请输入250Hz的LL值", min_value=0.0, max_value=500.0, value=120.0, step=0.1, format="%.1f")

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
        st.write(f"输入特征的形状: {features.shape}")

        # 使用加载的 StandardScaler 实例进行特征标准化
        features_scaled = scaler_X.transform(features)

        # 输出标准化后的特征的形状以进行验证
        st.write(f"标准化后的特征的形状: {features_scaled.shape}")

        # 输出模型系数的形状以进行验证
        st.write(f"模型系数的形状: {model.coef_.shape}")

        # 使用模型进行预测
        try:
            prediction = model.predict(features_scaled)
        except ValueError as e:
            st.error(f"预测时发生错误：{str(e)}")
            return

        # 输出预测结果的形状以进行验证
        st.write(f"预测结果的形状: {prediction.shape}")

        # 显示预测结果
        st.write("预测结果如下：")
        labels = [
            '体脂率', '肌肉', '骨骼肌', '水分率百分比', '蛋白质', '无机盐', '内脏脂肪', 
            '基础代谢', '节段肌肉左臂', '节段肌肉右臂', '节段肌肉躯干', '节段肌肉左腿', '节段肌肉右腿', '骨盐量', '腰臀比'
        ]
        
        # 计算水分含量
        water_percentage = prediction[0, 3]  # 获取水分率百分比
        water_content = weight * (water_percentage / 100)  # 计算水分含量
        
        # 计算 BMI
        bmi = weight / ((height / 100) ** 2)  # 身高单位为 cm，转换为 m
        
        # 添加 "水分含量" 和 "BMI" 标签
        labels_with_units = labels + ['水分含量', 'BMI']

        # 将水分含量和 BMI 添加到预测结果中
        predictions_with_water_and_bmi = np.hstack((prediction[0], [water_content, bmi]))

        # 定义每个标签的单位
        units = [
            '%', 'kg', 'kg', '%', 'kg', 'kg', '',  # 内脏脂肪没有单位
            '', '', '', '', '', '', 'kg', ''      # 腰臀比没有单位
        ]

        for i, (label, unit) in enumerate(zip(labels_with_units, units + ['kg', ''])):
            # 根据标签添加单位
            if unit:
                st.write(f"{label}: {predictions_with_water_and_bmi[i]:.2f} {unit}")
            else:
                st.write(f"{label}: {predictions_with_water_and_bmi[i]:.2f}")

if __name__ == "__main__":
    main()