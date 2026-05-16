use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use clap::{Args, Parser, Subcommand, ValueEnum};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const DEFAULT_GOOGLE_MODEL: &str = "gemma-4-31b-it";
const DEFAULT_GOOGLE_API: &str = "streamGenerateContent";
const DEFAULT_GOOGLE_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

#[derive(Debug, Parser)]
#[command(name = "gemma-agent-harness")]
#[command(about = "Rust LLM agent harness for Gemma-backed robot flows.")]
struct Cli {
    #[arg(long, default_value_t = ProviderKind::Google)]
    provider: ProviderKind,

    #[arg(long, default_value = DEFAULT_GOOGLE_MODEL)]
    model: String,

    #[arg(long)]
    env_file: Option<PathBuf>,

    #[arg(long)]
    instructions: Option<String>,

    #[arg(long, default_value_t = 500)]
    max_output_tokens: u32,

    #[arg(long, default_value_t = 4)]
    max_tool_rounds: usize,

    #[arg(long)]
    tool_config: Option<PathBuf>,

    #[arg(long, default_value = DEFAULT_GOOGLE_BASE_URL)]
    google_base_url: String,

    #[arg(long, default_value = DEFAULT_GOOGLE_API)]
    google_api: String,

    #[arg(long, default_value = "HIGH")]
    thinking_level: String,

    #[arg(long, default_value = "http://127.0.0.1:8765")]
    ios_bridge_url: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ProviderKind {
    Google,
    IosBridge,
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderKind::Google => write!(f, "google"),
            ProviderKind::IosBridge => write!(f, "ios-bridge"),
        }
    }
}

#[derive(Debug, Subcommand)]
enum Commands {
    Prompt(PromptArgs),
    Repl(ReplArgs),
    VoiceBot(VoiceBotArgs),
}

#[derive(Debug, Args)]
struct PromptArgs {
    #[arg(value_name = "TEXT")]
    text: Vec<String>,

    #[arg(long = "text-file")]
    text_files: Vec<PathBuf>,

    #[arg(long = "image")]
    images: Vec<PathBuf>,

    #[arg(long = "audio")]
    audio: Vec<PathBuf>,

    #[arg(long = "part", value_name = "PATH:MIME")]
    typed_parts: Vec<String>,

    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct ReplArgs {
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct VoiceBotArgs {
    #[arg(long, default_value = "~/gemma4-robot/voice-chat")]
    base_dir: String,

    #[arg(long, default_value = "~/gemma4-robot/kiosk/status.json")]
    status_file: String,

    #[arg(long, default_value_t = 23)]
    button_gpio: u32,

    #[arg(long, default_value_t = 25)]
    led_gpio: u32,

    #[arg(long)]
    playback_device: Option<String>,

    #[arg(long)]
    capture_device: Option<String>,

    #[arg(long, default_value_t = 16000)]
    sample_rate: u32,

    #[arg(long, default_value_t = 1)]
    channels: u32,

    #[arg(long, default_value_t = 0.35)]
    tap_reset_seconds: f64,

    #[arg(long, default_value = "")]
    startup_greeting: String,

    #[arg(
        long,
        default_value = "Respond briefly and conversationally to the user's spoken request."
    )]
    audio_prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Message {
    role: String,
    parts: Vec<Part>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Part {
    Text {
        text: String,
    },
    InlineData {
        mime_type: String,
        data: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        display_name: Option<String>,
    },
    FunctionCall {
        name: String,
        #[serde(default)]
        args: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    FunctionResponse {
        name: String,
        response: Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolDefinition {
    name: String,
    description: String,
    #[serde(default = "default_parameters_schema")]
    parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolCommand {
    #[serde(flatten)]
    definition: ToolDefinition,
    command: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolConfigFile {
    tools: Vec<ToolCommand>,
}

#[derive(Debug, Clone, Serialize)]
struct ToolCallRecord {
    name: String,
    args: Value,
    response: Value,
}

#[derive(Debug, Clone, Serialize)]
struct AgentRun {
    text: String,
    messages: Vec<Message>,
    tool_calls: Vec<ToolCallRecord>,
}

#[derive(Debug, Clone)]
struct ProviderRequest {
    model: String,
    instructions: Option<String>,
    messages: Vec<Message>,
    tools: Vec<ToolDefinition>,
    max_output_tokens: u32,
}

#[derive(Debug, Clone)]
struct ProviderResponse {
    message: Message,
}

trait LlmProvider {
    fn generate(&self, request: &ProviderRequest) -> Result<ProviderResponse>;
}

struct GoogleProvider {
    client: Client,
    api_key: String,
    base_url: String,
    api: String,
    thinking_level: String,
}

struct IosBridgeProvider {
    client: Client,
    base_url: String,
}

struct ToolRegistry {
    commands: HashMap<String, ToolCommand>,
}

struct Agent {
    provider: Box<dyn LlmProvider>,
    model: String,
    instructions: Option<String>,
    tools: ToolRegistry,
    max_output_tokens: u32,
    max_tool_rounds: usize,
    messages: Vec<Message>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    load_env_file(cli.env_file.as_deref())?;
    let tools = ToolRegistry::from_path(cli.tool_config.as_deref())?;
    let provider = make_provider(&cli)?;

    let mut agent = Agent {
        provider,
        model: cli.model.clone(),
        instructions: cli.instructions.clone(),
        tools,
        max_output_tokens: cli.max_output_tokens,
        max_tool_rounds: cli.max_tool_rounds,
        messages: Vec::new(),
    };

    match &cli.command {
        Commands::Prompt(args) => run_prompt(&mut agent, args),
        Commands::Repl(args) => run_repl(&mut agent, args),
        Commands::VoiceBot(args) => run_voice_bot(&mut agent, args),
    }
}

fn make_provider(cli: &Cli) -> Result<Box<dyn LlmProvider>> {
    let client = Client::builder()
        .timeout(Duration::from_secs(180))
        .build()
        .context("failed to create HTTP client")?;

    match cli.provider {
        ProviderKind::Google => {
            let api_key = std::env::var("GEMINI_API_KEY")
                .context("GEMINI_API_KEY is not set; put it in .env or the process environment")?;
            Ok(Box::new(GoogleProvider {
                client,
                api_key,
                base_url: cli.google_base_url.trim_end_matches('/').to_string(),
                api: cli.google_api.clone(),
                thinking_level: cli.thinking_level.clone(),
            }))
        }
        ProviderKind::IosBridge => Ok(Box::new(IosBridgeProvider {
            client,
            base_url: cli.ios_bridge_url.trim_end_matches('/').to_string(),
        })),
    }
}

fn run_prompt(agent: &mut Agent, args: &PromptArgs) -> Result<()> {
    let parts = prompt_parts(args)?;
    let run = agent.run(parts)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&run)?);
    } else {
        println!("{}", run.text);
    }
    Ok(())
}

fn run_repl(agent: &mut Agent, args: &ReplArgs) -> Result<()> {
    eprintln!("Gemma agent REPL. Empty line exits.");
    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush()?;
        let mut line = String::new();
        let n = stdin.read_line(&mut line)?;
        if n == 0 || line.trim().is_empty() {
            break;
        }
        let run = agent.run(vec![Part::Text {
            text: line.trim().to_string(),
        }])?;
        if args.json {
            println!("{}", serde_json::to_string_pretty(&run)?);
        } else {
            println!("{}", run.text);
        }
    }
    Ok(())
}

fn prompt_parts(args: &PromptArgs) -> Result<Vec<Part>> {
    let mut parts = Vec::new();
    if !args.text.is_empty() {
        parts.push(Part::Text {
            text: args.text.join(" "),
        });
    }

    for path in &args.text_files {
        parts.push(Part::Text {
            text: fs::read_to_string(path)
                .with_context(|| format!("failed to read text file {}", path.display()))?,
        });
    }

    for path in &args.images {
        parts.push(inline_file_part(path, infer_image_mime(path)?)?);
    }

    for path in &args.audio {
        parts.push(inline_file_part(path, infer_audio_mime(path)?)?);
    }

    for typed in &args.typed_parts {
        let (path, mime) = typed
            .rsplit_once(':')
            .ok_or_else(|| anyhow!("--part must be PATH:MIME, got {typed:?}"))?;
        parts.push(inline_file_part(Path::new(path), mime)?);
    }

    if parts.is_empty() {
        bail!("provide text, --text-file, --image, --audio, or --part PATH:MIME");
    }
    Ok(parts)
}

impl Agent {
    fn run(&mut self, user_parts: Vec<Part>) -> Result<AgentRun> {
        self.messages.push(Message {
            role: "user".to_string(),
            parts: user_parts,
        });

        let mut all_tool_calls = Vec::new();
        let mut final_text = String::new();

        for _round in 0..=self.max_tool_rounds {
            let request = ProviderRequest {
                model: self.model.clone(),
                instructions: self.instructions.clone(),
                messages: self.messages.clone(),
                tools: self.tools.definitions(),
                max_output_tokens: self.max_output_tokens,
            };
            let response = self.provider.generate(&request)?;
            final_text = text_from_parts(&response.message.parts);
            let function_calls = function_calls_from_parts(&response.message.parts);
            self.messages.push(response.message);

            if function_calls.is_empty() {
                return Ok(AgentRun {
                    text: final_text,
                    messages: self.messages.clone(),
                    tool_calls: all_tool_calls,
                });
            }

            for call in function_calls {
                let tool_response = self.tools.call(&call)?;
                all_tool_calls.push(ToolCallRecord {
                    name: call.name.clone(),
                    args: call.args.clone(),
                    response: tool_response.clone(),
                });
                self.messages.push(Message {
                    // The Gemini REST examples append ordinary function responses as a user turn.
                    role: "user".to_string(),
                    parts: vec![Part::FunctionResponse {
                        name: call.name,
                        response: tool_response,
                    }],
                });
            }
        }

        Ok(AgentRun {
            text: final_text,
            messages: self.messages.clone(),
            tool_calls: all_tool_calls,
        })
    }
}

impl GoogleProvider {
    fn request_url(&self, model: &str) -> String {
        format!(
            "{}/models/{}:{}?key={}",
            self.base_url,
            urlencoding::encode(model),
            self.api,
            urlencoding::encode(&self.api_key),
        )
    }
}

impl LlmProvider for GoogleProvider {
    fn generate(&self, request: &ProviderRequest) -> Result<ProviderResponse> {
        let mut body = json!({
            "contents": request.messages.iter().map(google_content).collect::<Vec<_>>(),
            "generationConfig": {
                "maxOutputTokens": request.max_output_tokens,
                "thinkingConfig": {
                    "thinkingLevel": self.thinking_level,
                },
            },
        });

        if let Some(instructions) = &request.instructions {
            body["systemInstruction"] = json!({
                "parts": [{"text": instructions}]
            });
        }

        if !request.tools.is_empty() {
            body["tools"] = json!([{
                "functionDeclarations": request.tools.iter().map(google_tool).collect::<Vec<_>>()
            }]);
        }

        let response = self
            .client
            .post(self.request_url(&request.model))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .context("Google generateContent request failed")?;

        let status = response.status();
        let text = response
            .text()
            .context("failed reading Google response body")?;
        if !status.is_success() {
            bail!("Google API error {status}: {text}");
        }

        parse_google_response(&text)
    }
}

impl LlmProvider for IosBridgeProvider {
    fn generate(&self, request: &ProviderRequest) -> Result<ProviderResponse> {
        let prompt = request
            .messages
            .iter()
            .map(message_to_prompt_text)
            .collect::<Vec<_>>()
            .join("\n\n");
        let body = json!({
            "prompt": prompt,
            "max_tokens": request.max_output_tokens,
            "timeout": 300,
        });
        let response = self
            .client
            .post(format!("{}/generate", self.base_url))
            .json(&body)
            .send()
            .context("iOS bridge request failed")?;
        let status = response.status();
        let value: Value = response
            .json()
            .context("failed reading iOS bridge response")?;
        if !status.is_success() {
            bail!("iOS bridge error {status}: {value}");
        }
        Ok(ProviderResponse {
            message: Message {
                role: "model".to_string(),
                parts: vec![Part::Text {
                    text: value
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                }],
            },
        })
    }
}

impl ToolRegistry {
    fn from_path(path: Option<&Path>) -> Result<Self> {
        let Some(path) = path else {
            return Ok(Self {
                commands: HashMap::new(),
            });
        };
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read tool config {}", path.display()))?;
        let value: Value = serde_json::from_str(&text)
            .with_context(|| format!("tool config is not valid JSON: {}", path.display()))?;
        let commands: Vec<ToolCommand> = if value.is_array() {
            serde_json::from_value(value)?
        } else {
            serde_json::from_value::<ToolConfigFile>(value)?.tools
        };
        Ok(Self {
            commands: commands
                .into_iter()
                .map(|command| (command.definition.name.clone(), command))
                .collect(),
        })
    }

    fn definitions(&self) -> Vec<ToolDefinition> {
        self.commands
            .values()
            .map(|command| command.definition.clone())
            .collect()
    }

    fn call(&self, call: &FunctionCall) -> Result<Value> {
        let Some(tool) = self.commands.get(&call.name) else {
            return Ok(json!({
                "ok": false,
                "error": format!("unknown tool: {}", call.name),
            }));
        };
        if tool.command.is_empty() {
            return Ok(json!({
                "ok": false,
                "error": format!("tool {} has an empty command", call.name),
            }));
        }

        let mut child = Command::new(&tool.command[0])
            .args(&tool.command[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("failed to start tool {}", call.name))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(
                serde_json::to_string(&json!({
                    "name": call.name,
                    "args": call.args,
                }))?
                .as_bytes(),
            )?;
        }

        let output = child
            .wait_with_output()
            .with_context(|| format!("tool {} failed while waiting", call.name))?;
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let parsed_stdout =
            serde_json::from_str::<Value>(&stdout).unwrap_or_else(|_| json!(stdout));

        Ok(json!({
            "ok": output.status.success(),
            "exit_code": output.status.code(),
            "result": parsed_stdout,
            "stderr": stderr,
        }))
    }
}

#[derive(Debug, Clone)]
struct FunctionCall {
    name: String,
    args: Value,
}

fn function_calls_from_parts(parts: &[Part]) -> Vec<FunctionCall> {
    parts
        .iter()
        .filter_map(|part| match part {
            Part::FunctionCall { name, args, .. } => Some(FunctionCall {
                name: name.clone(),
                args: args.clone(),
            }),
            _ => None,
        })
        .collect()
}

fn text_from_parts(parts: &[Part]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            Part::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
        .trim()
        .to_string()
}

fn google_content(message: &Message) -> Value {
    json!({
        "role": message.role,
        "parts": message.parts.iter().map(google_part).collect::<Vec<_>>(),
    })
}

fn google_part(part: &Part) -> Value {
    match part {
        Part::Text { text } => json!({ "text": text }),
        Part::InlineData {
            mime_type,
            data,
            display_name,
        } => {
            let mut inline = json!({
                "mimeType": mime_type,
                "data": data,
            });
            if let Some(display_name) = display_name {
                inline["displayName"] = json!(display_name);
            }
            json!({ "inlineData": inline })
        }
        Part::FunctionCall {
            name,
            args,
            thought_signature,
        } => {
            let mut value = json!({
                "functionCall": {
                    "name": name,
                    "args": args,
                }
            });
            if let Some(signature) = thought_signature {
                value["thoughtSignature"] = json!(signature);
            }
            value
        }
        Part::FunctionResponse { name, response } => json!({
            "functionResponse": {
                "name": name,
                "response": response,
            }
        }),
    }
}

fn google_tool(tool: &ToolDefinition) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    })
}

fn parse_google_response(body: &str) -> Result<ProviderResponse> {
    let chunks = parse_google_chunks(body)?;
    let mut parts = Vec::new();
    for chunk in chunks {
        let Some(candidates) = chunk.get("candidates").and_then(Value::as_array) else {
            continue;
        };
        for candidate in candidates {
            let Some(content) = candidate.get("content") else {
                continue;
            };
            let Some(candidate_parts) = content.get("parts").and_then(Value::as_array) else {
                continue;
            };
            for part in candidate_parts {
                if let Some(parsed) = parse_google_part(part) {
                    parts.push(parsed);
                }
            }
        }
    }

    if parts.is_empty() {
        bail!("Google response had no text or function-call parts: {body}");
    }

    Ok(ProviderResponse {
        message: Message {
            role: "model".to_string(),
            parts,
        },
    })
}

fn parse_google_chunks(body: &str) -> Result<Vec<Value>> {
    if let Ok(value) = serde_json::from_str::<Value>(body) {
        return Ok(match value {
            Value::Array(items) => items,
            other => vec![other],
        });
    }

    let mut chunks = Vec::new();
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let json_text = trimmed
            .strip_prefix("data:")
            .map(str::trim)
            .unwrap_or(trimmed);
        if json_text == "[DONE]" {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<Value>(json_text) {
            chunks.push(value);
        }
    }

    if chunks.is_empty() {
        bail!("Google response was neither JSON nor parseable SSE/JSONL: {body}");
    }
    Ok(chunks)
}

fn parse_google_part(value: &Value) -> Option<Part> {
    if value.get("thought").and_then(Value::as_bool) == Some(true)
        && value.get("functionCall").is_none()
    {
        return None;
    }
    if let Some(text) = value.get("text").and_then(Value::as_str) {
        return Some(Part::Text {
            text: text.to_string(),
        });
    }
    if let Some(call) = value.get("functionCall") {
        let name = call.get("name")?.as_str()?.to_string();
        let args = call.get("args").cloned().unwrap_or_else(|| json!({}));
        let thought_signature = value
            .get("thoughtSignature")
            .and_then(Value::as_str)
            .map(ToString::to_string);
        return Some(Part::FunctionCall {
            name,
            args,
            thought_signature,
        });
    }
    None
}

fn message_to_prompt_text(message: &Message) -> String {
    let content = message
        .parts
        .iter()
        .map(|part| match part {
            Part::Text { text } => text.clone(),
            Part::InlineData {
                mime_type,
                display_name,
                ..
            } => format!(
                "[inline {mime_type} {}]",
                display_name.as_deref().unwrap_or("")
            ),
            Part::FunctionCall { name, args, .. } => format!("[tool call {name}: {args}]"),
            Part::FunctionResponse { name, response } => {
                format!("[tool response {name}: {response}]")
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("{}: {}", message.role, content)
}

fn inline_file_part(path: &Path, mime_type: &str) -> Result<Part> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let data = base64::engine::general_purpose::STANDARD.encode(bytes);
    Ok(Part::InlineData {
        mime_type: mime_type.to_string(),
        data,
        display_name: path
            .file_name()
            .map(|name| name.to_string_lossy().to_string()),
    })
}

fn infer_image_mime(path: &Path) -> Result<&'static str> {
    match extension(path).as_deref() {
        Some("jpg") | Some("jpeg") => Ok("image/jpeg"),
        Some("png") => Ok("image/png"),
        Some("webp") => Ok("image/webp"),
        Some("heic") => Ok("image/heic"),
        other => bail!(
            "cannot infer image MIME type for {} ({other:?})",
            path.display()
        ),
    }
}

fn infer_audio_mime(path: &Path) -> Result<&'static str> {
    match extension(path).as_deref() {
        Some("wav") => Ok("audio/wav"),
        Some("mp3") => Ok("audio/mpeg"),
        Some("m4a") | Some("mp4") => Ok("audio/mp4"),
        Some("flac") => Ok("audio/flac"),
        Some("ogg") => Ok("audio/ogg"),
        other => bail!(
            "cannot infer audio MIME type for {} ({other:?})",
            path.display()
        ),
    }
}

fn extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
}

fn default_parameters_schema() -> Value {
    json!({
        "type": "object",
        "properties": {},
    })
}

fn load_env_file(path: Option<&Path>) -> Result<()> {
    let path = match path {
        Some(path) => path.to_path_buf(),
        None => {
            let cwd = PathBuf::from(".env");
            if cwd.exists() {
                cwd
            } else {
                expand_tilde("~/gemma4-robot/.env")
            }
        }
    };
    if !path.exists() {
        return Ok(());
    }
    let text = fs::read_to_string(&path)
        .with_context(|| format!("failed to read env file {}", path.display()))?;
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || !line.contains('=') {
            continue;
        }
        let (key, value) = line.split_once('=').expect("contains '='");
        if std::env::var_os(key.trim()).is_none() {
            std::env::set_var(
                key.trim(),
                value.trim().trim_matches('"').trim_matches('\''),
            );
        }
    }
    Ok(())
}

fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(path)
}

fn run_voice_bot(agent: &mut Agent, args: &VoiceBotArgs) -> Result<()> {
    let base_dir = expand_tilde(&args.base_dir);
    let status_file = expand_tilde(&args.status_file);
    let recordings_dir = base_dir.join("recordings");
    fs::create_dir_all(&recordings_dir)?;
    if let Some(parent) = status_file.parent() {
        fs::create_dir_all(parent)?;
    }

    let capture = args
        .capture_device
        .clone()
        .unwrap_or_else(|| "default".to_string());
    let mut gpio = VoiceGpio::new(args.button_gpio, args.led_gpio)?;
    let mut status = VoiceStatus::new(status_file);
    status.write("idle", "", &args.startup_greeting, "")?;
    gpio.led(false)?;
    eprintln!(
        "Ready. Hold GPIO {} to record; release to send audio to Gemma.",
        args.button_gpio
    );

    let mut recording: Option<(Child, PathBuf, Instant, usize)> = None;
    let mut previous_pressed = false;
    let mut turn = 0usize;

    loop {
        let pressed = gpio.button_pressed()?;
        if pressed && !previous_pressed {
            turn += 1;
            let path = recordings_dir.join(format!("turn-{turn:03}-{}.wav", compact_timestamp()));
            let sample_rate = args.sample_rate.to_string();
            let channels = args.channels.to_string();
            let child = Command::new("arecord")
                .args([
                    "-q",
                    "-D",
                    &capture,
                    "-f",
                    "S16_LE",
                    "-r",
                    &sample_rate,
                    "-c",
                    &channels,
                ])
                .arg(&path)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .context("failed to start arecord")?;
            gpio.led(true)?;
            status.write_turn(turn, "recording", "Recording audio", "", "")?;
            recording = Some((child, path, Instant::now(), turn));
        } else if !pressed && previous_pressed {
            if let Some((mut child, path, started, turn_index)) = recording.take() {
                let elapsed = started.elapsed();
                let _ = child.kill();
                let _ = child.wait();
                if elapsed.as_secs_f64() < args.tap_reset_seconds {
                    agent.messages.clear();
                    status.write_turn(turn_index, "reset", "", "", "")?;
                    gpio.led(false)?;
                } else {
                    status.write_turn(turn_index, "thinking", "Audio input", "", "")?;
                    let result = process_audio_turn(agent, &args.audio_prompt, &path)
                        .with_context(|| format!("audio turn failed for {}", path.display()));
                    match result {
                        Ok(run) => {
                            status.write_turn(turn_index, "idle", "Audio input", &run.text, "")?;
                            eprintln!("Assistant: {}", run.text);
                        }
                        Err(error) => {
                            status.write_turn(
                                turn_index,
                                "error",
                                "Audio input",
                                "",
                                &error.to_string(),
                            )?;
                            eprintln!("{error:?}");
                        }
                    }
                    gpio.led(false)?;
                }
            }
        }
        previous_pressed = pressed;
        thread::sleep(Duration::from_millis(25));
    }
}

fn process_audio_turn(agent: &mut Agent, audio_prompt: &str, path: &Path) -> Result<AgentRun> {
    let parts = vec![
        Part::Text {
            text: audio_prompt.to_string(),
        },
        inline_file_part(path, infer_audio_mime(path)?)?,
    ];
    agent.run(parts)
}

fn compact_timestamp() -> String {
    chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string()
}

struct VoiceStatus {
    path: PathBuf,
}

impl VoiceStatus {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }

    fn write(&mut self, state: &str, input: &str, output: &str, error: &str) -> Result<()> {
        self.write_turn(0, state, input, output, error)
    }

    fn write_turn(
        &mut self,
        turn: usize,
        state: &str,
        input: &str,
        output: &str,
        error: &str,
    ) -> Result<()> {
        let payload = json!({
            "mode": "voice",
            "state": state,
            "turn": turn,
            "input": input,
            "output": output,
            "error": error,
            "updated_at": chrono::Utc::now().to_rfc3339(),
        });
        let tmp = self.path.with_extension("json.tmp");
        fs::write(&tmp, serde_json::to_string_pretty(&payload)? + "\n")?;
        fs::rename(tmp, &self.path)?;
        Ok(())
    }
}

struct VoiceGpio {
    button_value: PathBuf,
    led_value: PathBuf,
}

impl VoiceGpio {
    fn new(button_pin: u32, led_pin: u32) -> Result<Self> {
        export_gpio(button_pin, "in")?;
        export_gpio(led_pin, "out")?;
        Ok(Self {
            button_value: PathBuf::from(format!("/sys/class/gpio/gpio{button_pin}/value")),
            led_value: PathBuf::from(format!("/sys/class/gpio/gpio{led_pin}/value")),
        })
    }

    fn button_pressed(&self) -> Result<bool> {
        let value = fs::read_to_string(&self.button_value)
            .with_context(|| format!("failed to read {}", self.button_value.display()))?;
        Ok(value.trim() == "0")
    }

    fn led(&mut self, on: bool) -> Result<()> {
        fs::write(&self.led_value, if on { "1" } else { "0" })
            .with_context(|| format!("failed to write {}", self.led_value.display()))?;
        Ok(())
    }
}

fn export_gpio(pin: u32, direction: &str) -> Result<()> {
    let gpio_dir = PathBuf::from(format!("/sys/class/gpio/gpio{pin}"));
    if !gpio_dir.exists() {
        fs::write("/sys/class/gpio/export", pin.to_string())
            .with_context(|| format!("failed to export GPIO {pin}"))?;
        thread::sleep(Duration::from_millis(100));
    }
    fs::write(gpio_dir.join("direction"), direction)
        .with_context(|| format!("failed to set GPIO {pin} direction"))?;
    Ok(())
}
