根据网络环境替换sources.list文件，或者删除Dockerfile中的
```
ADD sources.list /etc/apt/sources.list
```

构建镜像并运行
```
docker build -t fairdace .
docker run -itd -p 5000:5000 fairface
```

