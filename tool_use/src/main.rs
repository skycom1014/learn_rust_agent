use async_openai::{Client, config::OpenAIConfig};
use serde_json::{Value, json};
use std::{
    error::Error,
    io::{self, Write},
    path::{Component, Path, PathBuf},
    fs,
};
use tokio::{
    process::Command,
    time::{Duration, timeout},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let env_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(".env");
    dotenvy::from_path(&env_path).expect(".env 文件加载失败");

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY 未设置");
    let base_url = std::env::var("OPENAI_API_BASE_URL").ok();
    let model_name = std::env::var("MODEL_NAME").expect("MODEL_NAME 未设置");

    let mut config = OpenAIConfig::default().with_api_key(api_key);
    if let Some(url) = base_url {
        config = config.with_api_base(url);
    }
    let client = Client::with_config(config);

    let system_text = format!(
        "You are a coding agent at {}. Use bash to solve tasks. Act, don't explain.",
        std::env::current_dir()?.display()
    );
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }
        }
    },{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "limit": { "type": "integer" }
                },
                "required": ["path"]
            }
        }
    },{
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }
        }
    },{
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old_text": { "type": "string" },
                    "new_text": { "type": "string" }
                },
                "required": ["path", "old_text", "new_text"]
            }
        }
    }]);

    let mut history: Vec<Value> = vec![json!({"role": "system", "content": system_text})];
    loop {
        print!("s01 >> ");
        io::stdout().flush()?;
        let mut query = String::new();
        if io::stdin().read_line(&mut query).is_err() {
            break;
        }
        let query = query.trim().to_string();
        if query.is_empty() || matches!(query.as_str(), "q" | "exit") {
            break;
        }
        history.push(json!({"role": "user", "content": query}));
        agent_loop(&client, &model_name, &tools, &mut history).await?;
        println!();
    }

    Ok(())
}

async fn agent_loop(
    client: &Client<OpenAIConfig>,
    model: &str,
    tools: &Value,
    messages: &mut Vec<Value>,
) -> Result<(), Box<dyn Error>> {
    loop {
        let request = json!({
            "model": model,
            "messages": messages,
            "tools": tools,
            "max_tokens": 8000
        });

        let response: Value = client.chat().create_byot(request).await?;
        let message = response["choices"][0]["message"].clone();
        let tool_calls = message
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .cloned();
        messages.push(message.clone());

        if let Some(calls) = tool_calls {
            if calls.is_empty() {
                print_assistant_text(&message);
                return Ok(());
            }
            for call in calls {
                if call.get("type").and_then(|v| v.as_str()) != Some("function") {
                    continue;
                }
                let function = &call["function"];
                let name = function["name"].as_str().unwrap_or("");
                let args_raw = function["arguments"].as_str().unwrap_or("{}");
                let args: Value = serde_json::from_str(args_raw).unwrap_or_else(|_| json!({}));
                let output = match name {
                    "bash" => {
                        let command = args["command"].as_str().unwrap_or("");
                        if command.is_empty() {
                            "Error: command is empty".to_string()
                        } else {
                            println!("$ {}", command);
                            run_bash(command).await
                        }
                    }
                    "read_file" => {
                        let path = args["path"].as_str().unwrap_or("");
                        let limit = args.get("limit").and_then(|v| v.as_u64()).map(|v| v as usize);
                        if path.is_empty() {
                            "Error: path is empty".to_string()
                        } else {
                            run_read(path, limit)
                        }
                    }
                    "write_file" => {
                        let path = args["path"].as_str().unwrap_or("");
                        let content = args["content"].as_str().unwrap_or("");
                        if path.is_empty() {
                            "Error: path is empty".to_string()
                        } else {
                            run_write(path, content)
                        }
                    }
                    "edit_file" => {
                        let path = args["path"].as_str().unwrap_or("");
                        let old_text = args["old_text"].as_str().unwrap_or("");
                        let new_text = args["new_text"].as_str().unwrap_or("");
                        if path.is_empty() {
                            "Error: path is empty".to_string()
                        } else {
                            run_edit(path, old_text, new_text)
                        }
                    }
                    _ => continue,
                };
                if name != "bash" {
                    println!(
                        "> {}: {}",
                        name,
                        output.chars().take(200).collect::<String>()
                    );
                } else {
                    println!("{}", output.chars().take(200).collect::<String>());
                }
                let tool_message = json!({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": output
                });
                messages.push(tool_message);
            }
        } else {
            print_assistant_text(&message);
            return Ok(());
        }
    }
}

#[derive(Clone)]
enum PathPart {
    Prefix(std::ffi::OsString),
    RootDir,
    Normal(std::ffi::OsString),
}

// 规范化路径，去除 . 并处理 ..，保持可比较的路径形态
fn normalize_path(path: &Path) -> PathBuf {
    let mut parts: Vec<PathPart> = Vec::new();
    for comp in path.components() {
        match comp {
            Component::Prefix(prefix) => {
                parts.clear();
                parts.push(PathPart::Prefix(prefix.as_os_str().to_os_string()));
            }
            Component::RootDir => {
                parts.push(PathPart::RootDir);
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if let Some(last) = parts.last() {
                    if matches!(last, PathPart::Normal(_)) {
                        parts.pop();
                    }
                }
            }
            Component::Normal(s) => {
                parts.push(PathPart::Normal(s.to_os_string()));
            }
        }
    }
    let mut out = PathBuf::new();
    for part in parts {
        match part {
            PathPart::Prefix(s) => out.push(s),
            PathPart::RootDir => out.push(std::path::MAIN_SEPARATOR.to_string()),
            PathPart::Normal(s) => out.push(s),
        }
    }
    out
}

// 限定路径必须在工作目录下，防止路径逃逸
fn safe_path(input: &str) -> Result<PathBuf, String> {
    let workdir = std::env::current_dir().map_err(|e| format!("Error: {}", e))?;
    let workdir_norm = normalize_path(&workdir);
    let candidate = workdir.join(input);
    let candidate_norm = normalize_path(&candidate);
    if candidate_norm.starts_with(&workdir_norm) {
        Ok(candidate_norm)
    } else {
        Err(format!("Error: Path escapes workspace: {}", input))
    }
}

// 读取文件内容，按行截断并限制最大输出长度
fn run_read(path: &str, limit: Option<usize>) -> String {
    match safe_path(path) {
        Ok(fp) => match fs::read_to_string(&fp) {
            Ok(text) => {
                let mut lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
                if let Some(l) = limit {
                    if l < lines.len() {
                        let remaining = lines.len() - l;
                        lines.truncate(l);
                        lines.push(format!("... ({} more lines)", remaining));
                    }
                }
                let mut out = lines.join("\n");
                if out.len() > 50000 {
                    out = out.chars().take(50000).collect();
                }
                out
            }
            Err(e) => format!("Error: {}", e),
        },
        Err(e) => e,
    }
}

// 写入文件内容，必要时创建父目录
fn run_write(path: &str, content: &str) -> String {
    match safe_path(path) {
        Ok(fp) => {
            if let Some(parent) = fp.parent() {
                if let Err(e) = fs::create_dir_all(parent) {
                    return format!("Error: {}", e);
                }
            }
            match fs::write(&fp, content) {
                Ok(_) => format!("Wrote {} bytes to {}", content.len(), path),
                Err(e) => format!("Error: {}", e),
            }
        }
        Err(e) => e,
    }
}

// 用新文本替换首次出现的旧文本
fn run_edit(path: &str, old_text: &str, new_text: &str) -> String {
    match safe_path(path) {
        Ok(fp) => match fs::read_to_string(&fp) {
            Ok(content) => {
                if !content.contains(old_text) {
                    return format!("Error: Text not found in {}", path);
                }
                let updated = content.replacen(old_text, new_text, 1);
                match fs::write(&fp, updated) {
                    Ok(_) => format!("Edited {}", path),
                    Err(e) => format!("Error: {}", e),
                }
            }
            Err(e) => format!("Error: {}", e),
        },
        Err(e) => e,
    }
}



async fn run_bash(command: &str) -> String {
    let dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"];
    if dangerous.iter().any(|d| command.contains(d)) {
        return "Error: Dangerous command blocked".to_string();
    }
    let mut cmd = Command::new("powershell");
    let ps_command = format!(
        "$enc = [System.Text.UTF8Encoding]::new($false); \
        [Console]::OutputEncoding = $enc; \
        [Console]::InputEncoding = $enc; \
        $OutputEncoding = $enc; \
        $PSDefaultParameterValues['Out-File:Encoding']='utf8'; \
        $PSDefaultParameterValues['Set-Content:Encoding']='utf8'; \
        $PSDefaultParameterValues['Add-Content:Encoding']='utf8'; \
        chcp 65001 > $null; {}",
        command
    );
    cmd.arg("-Command").arg(ps_command);
    match timeout(Duration::from_secs(120), cmd.output()).await {
        Ok(Ok(output)) => {
            let mut out = String::new();
            out.push_str(&String::from_utf8_lossy(&output.stdout));
            out.push_str(&String::from_utf8_lossy(&output.stderr));
            let out = out.trim().to_string();
            if out.is_empty() {
                "(no output)".to_string()
            } else if out.len() > 50000 {
                out.chars().take(50000).collect()
            } else {
                out
            }
        }
        Ok(Err(err)) => format!("Error: {}", err),
        Err(_) => "Error: Timeout (120s)".to_string(),
    }
}

fn print_assistant_text(message: &Value) {
    if let Some(content) = message.get("content").and_then(|v| v.as_str()) {
        if !content.is_empty() {
            println!("{}", content);
        } else {
            println!("响应为空");
        }
    } else {
        println!("响应为空");
    }
}
