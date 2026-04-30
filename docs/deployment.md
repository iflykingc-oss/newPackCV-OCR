# 部署文档

## 目录
- [Docker部署](#docker部署)
- [Kubernetes部署](#kubernetes部署)
- [环境配置](#环境配置)
- [健康检查](#健康检查)
- [故障排查](#故障排查)

## Docker部署

### 前置要求
- Docker 20.10+
- Docker Compose 2.0+

### 快速部署
```bash
cd deploy

# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f packcv
```

### 服务说明
| 服务 | 端口 | 说明 |
|------|------|------|
| packcv | 8000 | 主应用 |
| postgres | 5432 | PostgreSQL数据库 |
| redis | 6379 | Redis缓存 |
| minio | 9000/9001 | 对象存储+控制台 |
| nginx | 80/443 | 反向代理（可选） |

### 数据持久化
所有数据通过Docker卷持久化：
- `postgres_data` - 数据库数据
- `redis_data` - 缓存数据
- `minio_data` - 文件存储
- `packcv_data` - 应用数据

### 访问地址
- 应用API: http://localhost:8000
- MinIO控制台: http://localhost:9001
- API文档: http://localhost:8000/docs

## Kubernetes部署

### 前置要求
- Kubernetes 1.20+
- Helm 3.0+
- Ingress Controller

### 部署步骤
```bash
# 创建命名空间
kubectl apply -f deploy/k8s-deployment.yaml

# 检查Pod状态
kubectl get pods -n packcv

# 查看服务日志
kubectl logs -n packcv -l app=packcv-ocr

# 暴露服务（NodePort）
kubectl expose deployment packcv-ocr -n packcv --type=NodePort --port=8000
```

### 配置Secret
```bash
# 创建密钥
kubectl create secret generic packcv-secrets \
  --from-literal=DB_PASSWORD=your_password \
  --from-literal=REDIS_PASSWORD=your_password \
  --from-literal=OSS_ACCESS_KEY=your_key \
  --from-literal=OSS_SECRET_KEY=your_secret \
  -n packcv
```

### 配置PV（持久卷）
```yaml
# packcv-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: packcv-data-pvc
  namespace: packcv
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

## 环境配置

### 必需环境变量
```bash
# 数据库
DB_HOST=postgres
DB_PORT=5432
DB_NAME=packcv
DB_USER=postgres
DB_PASSWORD=your_secure_password

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=your_secure_password

# 对象存储
OSS_ENDPOINT=http://minio:9000
OSS_ACCESS_KEY=minioadmin
OSS_SECRET_KEY=minioadmin
OSS_BUCKET=packcv

# 日志
LOG_LEVEL=INFO
```

### 可选环境变量
```bash
# LLM配置
LLM_API_KEY=your_api_key
LLM_MODEL=gpt-4o
LLM_BASE_URL=https://api.openai.com/v1

# 性能调优
MAX_WORKERS=8
BATCH_SIZE=10
CACHE_TTL=3600
```

## 健康检查

### API健康检查
```bash
curl http://localhost:8000/api/v1/health
```

响应示例：
```json
{
    "status": "healthy",
    "timestamp": "2025-01-01T00:00:00Z",
    "components": {
        "database": "healthy",
        "redis": "healthy",
        "storage": "healthy"
    }
}
```

### Docker健康检查
```bash
docker inspect --format='{{.State.Health.Status}}' packcv-ocr
```

## 故障排查

### 服务无法启动
1. 检查日志：
   ```bash
   docker-compose logs packcv
   ```

2. 检查端口占用：
   ```bash
   netstat -tlnp | grep 8000
   ```

3. 检查依赖服务：
   ```bash
   docker-compose ps
   ```

### 数据库连接失败
1. 检查PostgreSQL状态：
   ```bash
   docker-compose logs postgres
   ```

2. 测试连接：
   ```bash
   docker-compose exec postgres psql -U postgres -d packcv
   ```

3. 检查网络：
   ```bash
   docker network inspect packcv_packcv-network
   ```

### OCR识别失败
1. 检查OCR引擎安装：
   ```bash
   docker-compose exec packcv tesseract --version
   ```

2. 检查模型文件：
   ```bash
   docker-compose exec packcv ls -la /app/models/
   ```

3. 查看详细日志：
   ```bash
   docker-compose logs -f packcv | grep OCR
   ```

### 性能问题
1. 检查资源使用：
   ```bash
   docker stats
   ```

2. 调整并发数：
   ```bash
   # 修改docker-compose.yml
   environment:
     - MAX_WORKERS=16
   ```

3. 启用缓存：
   ```bash
   docker-compose exec packcv python -c "from src.storage.cache import get_cache; c = get_cache(); c.clear('*')"
   ```

## 备份与恢复

### 数据库备份
```bash
# 备份
docker-compose exec postgres pg_dump -U postgres packcv > backup.sql

# 恢复
docker-compose exec -T postgres psql -U postgres packcv < backup.sql
```

### 对象存储备份
```bash
# 使用mc客户端
docker-compose run --rm minio-client mc mirror local/packcv /backup/
```

## 升级

### Docker Compose升级
```bash
# 拉取最新镜像
docker-compose pull

# 重启服务
docker-compose up -d
```

### 数据迁移
```bash
# 1. 停止服务
docker-compose down

# 2. 备份数据
docker run --rm -v packcv_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres.tar.gz /data

# 3. 更新代码
git pull

# 4. 重新启动
docker-compose up -d
```
