import numpy as np
import streamlit as st
from joblib import load
from sklearn.preprocessing import StandardScaler
from setuptools import setup
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'joblib'])

# Streamlit界面
def main():
    st.title('健康指标预测')

    # 创建文本输入框供用户输入特征数据
    weight = st.number_input("请输入体重 (kg)", min_value=0, max_value=150, value=65)
    height = st.number_input("请输入身高 (cm)", min_value=0, max_value=250, value=175)
    age = st.number_input("请输入年龄", min_value=0, max_value=120, value=25)
    ra5 = st.number_input("请输入5Hz的RA值", min_value=0, max_value=500, value=100)
    la5 = st.number_input("请输入5Hz的LA值", min_value=0, max_value=500, value=100)
    tr5 = st.number_input("请输入5Hz的TR值", min_value=0, max_value=500, value=50)
    rl5 = st.number_input("请输入5Hz的RL值", min_value=0, max_value=500, value=120)
    ll5 = st.number_input("请输入5Hz的LL值", min_value=0, max_value=500, value=120)
    ra50 = st.number_input("请输入50Hz的RA值", min_value=0, max_value=500, value=100)
    la50 = st.number_input("请输入50Hz的LA值", min_value=0, max_value=500, value=100)
    tr50 = st.number_input("请输入50Hz的TR值", min_value=0, max_value=500, value=50)
    rl50 = st.number_input("请输入50Hz的RL值", min_value=0, max_value=500, value=120)
    ll50 = st.number_input("请输入50Hz的LL值", min_value=0, max_value=500, value=120)
    ra250 = st.number_input("请输入250Hz的RA值", min_value=0, max_value=500, value=100)
    la250 = st.number_input("请输入250Hz的LA值", min_value=0, max_value=500, value=100)
    tr250 = st.number_input("请输入250Hz的TR值", min_value=0, max_value=500, value=50)
    rl250 = st.number_input("请输入250Hz的RL值", min_value=0, max_value=500, value=120)
    ll250 = st.number_input("请输入250Hz的LL值", min_value=0, max_value=500, value=120)

    # 构建特征数组
    features = np.array([weight, height, age, ra5, la5, tr5, rl5, ll5, ra50, la50, tr50, rl50, ll50, ra250, la250, tr250, rl250, ll250]).reshape(1, -1)
    
    # 用户点击预测按钮
    if st.button('进行预测'):
        # 加载模型和标准化器
        model = load('E:\机器学习源文件\项目6\heu.joblib')
        scaler_X = load('E:\机器学习源文件\项目6\scaler_X.joblib')

        # 使用加载的 StandardScaler 实例进行特征标准化
        features_scaled = scaler_X.transform(features)

        # 使用模型进行预测
        prediction = model.predict(features_scaled)

        # 显示预测结果
        st.write("预测结果如下：")
        labels = ['体脂率1', '肌肉1', '骨骼肌1', '水分量1', '蛋白质1', '无机盐1', '内脏脂肪1', '基础代谢1', '节段肌肉左臂1', '节段肌肉右臂1', '节段肌肉躯干1', '节段肌肉左腿1', '节段肌肉右腿1', '骨量1', '腰臀比1', '浮肿评估1']
        # 检查 prediction 的形状
        if len(prediction) > 0:
            for i, label in enumerate(labels):
                # 直接访问 prediction 的元素
                st.write(f"{label}: {prediction[0][i]:.2f}")
        else:
            st.write("预测结果为空，请检查模型和输入数据。")

if __name__ == "__main__":
    main()