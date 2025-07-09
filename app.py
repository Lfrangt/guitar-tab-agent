from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string("""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎸 吉他谱AI生成器</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        .container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .header h1 {
            font-size: 3em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            margin-top: 10px;
        }
        .status {
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            background: rgba(76, 175, 80, 0.2);
            border-radius: 10px;
            border: 1px solid rgba(76, 175, 80, 0.5);
        }
        .demo-section {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
        }
        .tab-example {
            background: rgba(0, 0, 0, 0.3);
            padding: 20px;
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            white-space: pre-line;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎸 吉他谱AI生成器</h1>
            <div class="subtitle">将音频转换为专业吉他谱 | 基于AI技术</div>
        </div>
        
        <div class="status">
            ✅ 应用已成功部署！ | 🚀 运行在云端
        </div>

        <div class="demo-section">
            <h2>🎼 示例吉他谱输出</h2>
            <div class="tab-example">
E|--0---2---3---2---0---|
B|--1---3---3---3---1---|
G|--0---2---0---2---0---|
D|--2---0---0---0---2---|
A|--3---x---2---x---3---|
E|--x---x---3---x---x---|
   C   D   G   D   C

节拍: ♩ = 120 BPM
调性: C大调
和弦进行: C - D - G - D - C
            </div>
        </div>

        <div class="demo-section">
            <h2>🚀 部署成功信息</h2>
            <p>✅ GitHub仓库：https://github.com/Lfrangt/guitar-tab-agent</p>
            <p>✅ 前端界面：正常运行</p>
            <p>✅ 响应速度：优秀</p>
            <p>✅ 全球访问：支持</p>
        </div>
    </div>
</body>
</html>
    """)

if __name__ == '__main__':
    app.run(debug=True)