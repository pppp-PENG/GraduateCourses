#!/bin/bash

# 构建Docker镜像
echo "构建Docker镜像..."
docker build -t news-topic-classifier .

# 测试本地运行
echo "测试本地运行..."
docker run -d -p 5724:5724 --name news-classifier news-topic-classifier

echo "等待服务启动..."
sleep 10

# 测试API
echo "测试API服务..."
python test_api.py

# 停止容器
echo "停止测试容器..."
docker stop news-classifier
docker rm news-classifier

echo "部署完成！"