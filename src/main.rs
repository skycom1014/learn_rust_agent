use async_openai::{Client, config::OpenAIConfig};
use serde_json::{Value, json};
use std::{
    error::Error,
    io::{self, Write},
    path::Path,
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
                if name != "bash" {
                    continue;
                }
                let args_raw = function["arguments"].as_str().unwrap_or("{}");
                let args: Value = serde_json::from_str(args_raw).unwrap_or_else(|_| json!({}));
                let command = args["command"].as_str().unwrap_or("");
                if command.is_empty() {
                    continue;
                }
                println!("$ {}", command);
                let output = run_bash(command).await;
                println!("{}", output.chars().take(200).collect::<String>());
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
