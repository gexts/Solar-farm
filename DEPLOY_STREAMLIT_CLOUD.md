# Streamlit Community Cloud 部署说明

本项目已经整理为可部署的 Streamlit 应用。

推荐入口文件：

- `app.py`

依赖文件：

- `requirements.txt`

主要步骤：

1. 先把当前项目代码上传到一个 GitHub 仓库。
2. 打开 Streamlit Community Cloud。
3. 选择 `New app`。
4. 选择你的 GitHub 仓库、分支和入口文件 `app.py`。
5. 点击 `Deploy`。
6. 部署成功后，平台会分配一个公网 URL，通常形如：
   `https://<your-app-name>.streamlit.app`

部署时至少需要这些文件：

- `app.py`
- `streamlit_app_v2.py`
- `requirements.txt`
- `solar-farm-design/`

说明：

- 该公网地址由 Streamlit 平台在部署时生成，无法在本地预先固定。
- 如果后续修改代码，只需推送到 GitHub，Streamlit 应用会自动更新或手动重部署。
