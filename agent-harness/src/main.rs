use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use clap::{Args, Parser, Subcommand, ValueEnum};
use reqwest::blocking::{Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const DEFAULT_GOOGLE_MODEL: &str = "gemma-4-31b-it";
const DEFAULT_GOOGLE_API: &str = "streamGenerateContent";
const DEFAULT_GOOGLE_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_IOS_BRIDGE_URL: &str = "http://127.0.0.1:8765";
const IOS_GENERATE_BINARY_MAGIC: &[u8] = b"G4GEN01";

#[derive(Debug, Parser)]
#[command(name = "gemma-agent-harness")]
#[command(about = "Rust LLM agent harness for Gemma-backed robot flows.")]
struct Cli {
    #[arg(long, env = "GEMMA_AGENT_PROVIDER", default_value_t = ProviderKind::Google)]
    provider: ProviderKind,

    #[arg(long, env = "GEMMA_AGENT_MODEL", default_value = DEFAULT_GOOGLE_MODEL)]
    model: String,

    #[arg(long, env = "GEMMA_AGENT_ENV_FILE")]
    env_file: Option<PathBuf>,

    #[arg(long, env = "GEMMA_AGENT_INSTRUCTIONS")]
    instructions: Option<String>,

    #[arg(long, env = "GEMMA_AGENT_INSTRUCTIONS_FILE")]
    instructions_file: Option<PathBuf>,

    #[arg(long, env = "GEMMA_AGENT_MAX_OUTPUT_TOKENS", default_value_t = 500)]
    max_output_tokens: u32,

    #[arg(long, env = "GEMMA_AGENT_MAX_TOOL_ROUNDS", default_value_t = 4)]
    max_tool_rounds: usize,

    #[arg(long, env = "GEMMA_AGENT_TOOL_CONFIG")]
    tool_config: Option<PathBuf>,

    #[arg(long, env = "GEMMA_GOOGLE_BASE_URL", default_value = DEFAULT_GOOGLE_BASE_URL)]
    google_base_url: String,

    #[arg(long, env = "GEMMA_GOOGLE_API", default_value = DEFAULT_GOOGLE_API)]
    google_api: String,

    #[arg(long, env = "GEMMA_GOOGLE_THINKING_LEVEL", default_value = "HIGH")]
    thinking_level: String,

    #[arg(long, env = "GEMMA_IOS_BRIDGE_URL", default_value = DEFAULT_IOS_BRIDGE_URL)]
    ios_bridge_url: String,

    #[arg(long, env = "GEMMA_IOS_BRIDGE_TIMEOUT_SECONDS", default_value_t = 300)]
    ios_bridge_timeout_seconds: u64,

    #[arg(
        long,
        env = "GEMMA_IOS_BRIDGE_ALLOW_NOT_READY",
        default_value_t = false
    )]
    ios_bridge_allow_not_ready: bool,

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

    #[arg(long, default_value = "~/gemma4-robot/voice-chat/speech")]
    speech_dir: String,

    #[arg(long, default_value = "~/gemma4-robot/kiosk/status.json")]
    status_file: String,

    #[arg(long, default_value = "~/gemma4-robot/kiosk/sensors.json")]
    sensors_file: String,

    #[arg(long, default_value_t = 23)]
    button_gpio: u32,

    #[arg(long, default_value_t = 25)]
    led_gpio: u32,

    #[arg(long, default_value_t = ButtonSource::Gpio)]
    button_source: ButtonSource,

    #[arg(long, default_value_t = LedSource::Gpio)]
    led_source: LedSource,

    #[arg(long, default_value = "auto")]
    microbit_device: String,

    #[arg(long, default_value_t = 115200)]
    microbit_baud: u32,

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

    #[arg(long, default_value = "")]
    startup_agent_prompt: String,

    #[arg(long, default_value_t = TranscriptionProvider::None)]
    transcription_provider: TranscriptionProvider,

    #[arg(long, default_value_t = TtsProvider::Auto)]
    tts_provider: TtsProvider,

    #[arg(long)]
    tts_command: Option<String>,

    #[arg(long, default_value_t = 3900)]
    tts_max_chars: usize,

    #[arg(long, env = "GEMMA_IOS_TTS_URL")]
    iphone_tts_url: Option<String>,

    #[arg(long, env = "GEMMA_IOS_TTS_BACKEND", default_value = "piper-ryan-high")]
    iphone_tts_backend: String,

    #[arg(long, env = "GEMMA_IOS_TTS_VOICE", default_value = "")]
    iphone_tts_voice: String,

    #[arg(long, default_value_t = true)]
    fullscreen_terminal: bool,

    #[arg(
        long,
        default_value = "Respond briefly and conversationally to the user's spoken request. Do not use emojis."
    )]
    audio_prompt: String,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TranscriptionProvider {
    Auto,
    None,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum ButtonSource {
    Gpio,
    MicrobitSerial,
}

impl std::fmt::Display for ButtonSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ButtonSource::Gpio => write!(f, "gpio"),
            ButtonSource::MicrobitSerial => write!(f, "microbit-serial"),
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum LedSource {
    Gpio,
    None,
}

impl std::fmt::Display for LedSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LedSource::Gpio => write!(f, "gpio"),
            LedSource::None => write!(f, "none"),
        }
    }
}

impl std::fmt::Display for TranscriptionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TranscriptionProvider::Auto => write!(f, "auto"),
            TranscriptionProvider::None => write!(f, "none"),
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum TtsProvider {
    Auto,
    None,
    Command,
    Iphone,
}

impl std::fmt::Display for TtsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TtsProvider::Auto => write!(f, "auto"),
            TtsProvider::None => write!(f, "none"),
            TtsProvider::Command => write!(f, "command"),
            TtsProvider::Iphone => write!(f, "iphone"),
        }
    }
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
struct VoiceTurnStats {
    audio_bytes: u64,
    audio_seconds: f64,
    output_tokens_estimate: usize,
    elapsed_seconds: f64,
    tokens_per_second: f64,
}

#[derive(Debug, Clone)]
struct VoiceTurnRun {
    run: AgentRun,
    stats: VoiceTurnStats,
}

#[derive(Debug, Clone)]
struct ProviderRequest {
    model: String,
    instructions: Option<String>,
    messages: Vec<Message>,
    tools: Vec<ToolDefinition>,
    max_output_tokens: u32,
}

struct IosBridgeRequest {
    prompt: String,
    media: Vec<IosBridgeMedia>,
}

struct IosBridgeMedia {
    mime_type: String,
    bytes: Vec<u8>,
    display_name: Option<String>,
}

#[derive(Debug, Clone)]
struct ProviderResponse {
    message: Message,
}

trait LlmProvider {
    fn generate(
        &self,
        request: &ProviderRequest,
        on_text_delta: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<ProviderResponse>;
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
    timeout_seconds: u64,
    require_ready: bool,
}

#[derive(Debug, Deserialize)]
struct IosBridgeHealth {
    worker_connected: bool,
    #[serde(default)]
    worker_status: Value,
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
    let env_file = preparse_env_file_arg()
        .or_else(|| std::env::var_os("GEMMA_AGENT_ENV_FILE").map(PathBuf::from));
    load_env_file(env_file.as_deref())?;
    let cli = Cli::parse();
    let tools = ToolRegistry::from_path(cli.tool_config.as_deref())?;
    let provider = make_provider(&cli)?;

    let instructions = load_instructions(cli.instructions.clone(), cli.instructions_file.as_deref())?;

    let mut agent = Agent {
        provider,
        model: cli.model.clone(),
        instructions,
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

fn preparse_env_file_arg() -> Option<PathBuf> {
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--env-file" {
            return args.next().map(PathBuf::from);
        }
        let text = arg.to_string_lossy();
        if let Some(value) = text.strip_prefix("--env-file=") {
            return Some(PathBuf::from(value));
        }
    }
    None
}

fn make_provider(cli: &Cli) -> Result<Box<dyn LlmProvider>> {
    let http_timeout = match cli.provider {
        ProviderKind::Google => Duration::from_secs(180),
        ProviderKind::IosBridge => Duration::from_secs(cli.ios_bridge_timeout_seconds + 15),
    };
    let client = Client::builder()
        .timeout(http_timeout)
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
            timeout_seconds: cli.ios_bridge_timeout_seconds,
            require_ready: !cli.ios_bridge_allow_not_ready,
        })),
    }
}

fn load_instructions(inline: Option<String>, path: Option<&Path>) -> Result<Option<String>> {
    let file_text = match path {
        Some(path) => Some(
            fs::read_to_string(path)
                .with_context(|| format!("failed to read instructions file {}", path.display()))?,
        ),
        None => None,
    };
    Ok(match (inline, file_text) {
        (Some(inline), Some(file_text)) => Some(format!("{}\n\n{}", file_text.trim(), inline.trim())),
        (Some(inline), None) => Some(inline),
        (None, Some(file_text)) => Some(file_text),
        (None, None) => None,
    })
}

fn run_prompt(agent: &mut Agent, args: &PromptArgs) -> Result<()> {
    let parts = prompt_parts(args)?;
    let run = if args.json {
        agent.run(parts)?
    } else {
        let mut streamed = false;
        let run = {
            let mut stream = |delta: &str| -> Result<()> {
                if !delta.is_empty() {
                    streamed = true;
                    print!("{delta}");
                    io::stdout().flush()?;
                }
                Ok(())
            };
            agent.run_with_stream(parts, &mut stream)?
        };
        if streamed {
            println!();
        } else {
            println!("{}", run.text);
        }
        return Ok(());
    };
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
        let parts = vec![Part::Text {
            text: line.trim().to_string(),
        }];
        if args.json {
            let run = agent.run(parts)?;
            println!("{}", serde_json::to_string_pretty(&run)?);
        } else {
            let mut streamed = false;
            let run = {
                let mut stream = |delta: &str| -> Result<()> {
                    if !delta.is_empty() {
                        streamed = true;
                        print!("{delta}");
                        io::stdout().flush()?;
                    }
                    Ok(())
                };
                agent.run_with_stream(parts, &mut stream)?
            };
            if streamed {
                println!();
            } else {
                println!("{}", run.text);
            }
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
        let mut ignore_delta = |_delta: &str| Ok(());
        self.run_with_stream(user_parts, &mut ignore_delta)
    }

    fn run_with_stream(
        &mut self,
        user_parts: Vec<Part>,
        on_text_delta: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<AgentRun> {
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
            let response = self.provider.generate(&request, on_text_delta)?;
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
        let mut url = format!(
            "{}/models/{}:{}?key={}",
            self.base_url,
            urlencoding::encode(model),
            self.api,
            urlencoding::encode(&self.api_key),
        );
        if self.api.starts_with("stream") {
            url.push_str("&alt=sse");
        }
        url
    }
}

impl LlmProvider for GoogleProvider {
    fn generate(
        &self,
        request: &ProviderRequest,
        on_text_delta: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<ProviderResponse> {
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
        if !status.is_success() {
            let text = response
                .text()
                .context("failed reading Google error response body")?;
            bail!("Google API error {status}: {text}");
        }

        read_google_stream_response(response, on_text_delta)
    }
}

impl LlmProvider for IosBridgeProvider {
    fn generate(
        &self,
        request: &ProviderRequest,
        on_text_delta: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<ProviderResponse> {
        let health = self.health().context("failed to check iOS bridge health")?;
        if !health.worker_connected {
            bail!(
                "iOS bridge at {} has no connected iPhone worker; open Gemma Inference Server on the iPhone and tap Connect to ws://pi3:8765/worker",
                self.base_url
            );
        }
        if self.require_ready && health.runtime_ready() != Some(true) {
            bail!(
                "iPhone worker is connected but Gemma runtime is not ready: {}",
                health.status_summary()
            );
        }

        let bridge_request = bridge_request(request)?;
        let use_tool_shim = !request.tools.is_empty();
        let response = if bridge_request.media.is_empty() {
            let body = json!({
                "model": request.model,
                "prompt": bridge_request.prompt,
                "max_tokens": request.max_output_tokens,
                "timeout": self.timeout_seconds,
            });
            self.client
                .post(format!("{}/generate-stream", self.base_url))
                .json(&body)
                .send()
                .context("iOS bridge stream request failed")?
        } else {
            let body = pack_ios_generate_binary(
                &bridge_request.prompt,
                request.max_output_tokens,
                self.timeout_seconds,
                &bridge_request.media,
            )?;
            self.client
                .post(format!("{}/generate-media-stream", self.base_url))
                .header("Content-Type", "application/octet-stream")
                .body(body)
                .send()
                .context("iOS bridge media stream request failed")?
        };
        let status = response.status();
        if !status.is_success() {
            let value: Value = response
                .json()
                .context("failed reading iOS bridge error response")?;
            bail!(
                "iOS bridge error {status}: {}",
                bridge_error_message(&value)
            );
        }
        let mut streamed_text_to_status = false;
        let text = if use_tool_shim {
            let mut buffered_prefix = String::new();
            let mut decided_text = false;
            let mut shim_delta = |delta: &str| -> Result<()> {
                if decided_text {
                    streamed_text_to_status = true;
                    return on_text_delta(delta);
                }

                buffered_prefix.push_str(delta);
                let trimmed = buffered_prefix.trim_start();
                if trimmed.is_empty() {
                    return Ok(());
                }

                if trimmed.starts_with('{') || trimmed.starts_with('[') || trimmed.starts_with("```") {
                    return Ok(());
                }

                decided_text = true;
                streamed_text_to_status = true;
                on_text_delta(&buffered_prefix)
            };
            read_ios_bridge_stream_response(response, &mut shim_delta)?
        } else {
            read_ios_bridge_stream_response(response, on_text_delta)?
        };
        let parts = if use_tool_shim {
            ios_bridge_parts_from_text(&text, &request.tools, on_text_delta, streamed_text_to_status)?
        } else {
            vec![Part::Text {
                text: text.to_string(),
            }]
        };
        Ok(ProviderResponse {
            message: Message {
                role: "model".to_string(),
                parts,
            },
        })
    }
}

impl IosBridgeProvider {
    fn health(&self) -> Result<IosBridgeHealth> {
        let response = self
            .client
            .get(format!("{}/health", self.base_url))
            .send()
            .with_context(|| format!("failed to call {}/health", self.base_url))?;
        let status = response.status();
        let value: Value = response
            .json()
            .context("failed reading iOS bridge health response")?;
        if !status.is_success() {
            bail!(
                "iOS bridge health error {status}: {}",
                bridge_error_message(&value)
            );
        }
        serde_json::from_value(value).context("iOS bridge health response had unexpected shape")
    }
}

impl IosBridgeHealth {
    fn runtime_ready(&self) -> Option<bool> {
        self.worker_status
            .get("runtime_ready")
            .and_then(Value::as_bool)
    }

    fn status_summary(&self) -> String {
        let runtime = self
            .worker_status
            .get("runtime")
            .and_then(Value::as_str)
            .unwrap_or("unknown-runtime");
        let runtime_status = self
            .worker_status
            .get("runtime_status")
            .and_then(Value::as_str)
            .unwrap_or("no runtime_status reported");
        format!(
            "runtime={runtime}, ready={:?}, status={runtime_status}",
            self.runtime_ready()
        )
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
    let text = parts
        .iter()
        .filter_map(|part| match part {
            Part::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
        .trim()
        .to_string();
    clean_user_text(&text)
}

fn clean_user_text(text: &str) -> String {
    let mut cleaned = text.trim_start();
    while let Some(rest) = cleaned.strip_prefix(':') {
        cleaned = rest.trim_start();
    }
    cleaned.trim().to_string()
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

fn read_google_stream_response(
    response: Response,
    on_text_delta: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<ProviderResponse> {
    let mut reader = BufReader::new(response);
    let mut body = String::new();
    let mut chunks = Vec::new();

    loop {
        let mut line = String::new();
        let bytes = reader
            .read_line(&mut line)
            .context("failed reading Google response stream")?;
        if bytes == 0 {
            break;
        }
        body.push_str(&line);

        if let Some(chunk) = google_stream_chunk_from_line(&line) {
            emit_google_text_deltas(&chunk, on_text_delta)?;
            chunks.push(chunk);
        }
    }

    if chunks.is_empty() {
        parse_google_response(&body)
    } else {
        provider_response_from_google_chunks(chunks, &body)
    }
}

fn read_ios_bridge_stream_response(
    response: Response,
    on_text_delta: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<String> {
    let reader = BufReader::new(response);
    let mut text = String::new();

    for line in reader.lines() {
        let line = line.context("failed reading iOS bridge stream")?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let event: Value = serde_json::from_str(trimmed)
            .with_context(|| format!("iOS bridge stream sent invalid JSON line: {trimmed}"))?;
        match event.get("type").and_then(Value::as_str) {
            Some("token") => {
                let token = event.get("text").and_then(Value::as_str).unwrap_or("");
                if !token.is_empty() {
                    text.push_str(token);
                    on_text_delta(token)?;
                }
            }
            Some("done") => {
                return Ok(event
                    .get("text")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
                    .unwrap_or(text));
            }
            Some("error") => bail!(
                "iOS bridge stream error: {}",
                event
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown stream error")
            ),
            _ => {}
        }
    }

    bail!("iOS bridge stream ended without a final done event")
}

fn google_stream_chunk_from_line(line: &str) -> Option<Value> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed == "[" || trimmed == "]" || trimmed == "," {
        return None;
    }

    let json_text = trimmed
        .strip_prefix("data:")
        .map(str::trim)
        .unwrap_or(trimmed)
        .trim_end_matches(',');
    if json_text == "[DONE]" || json_text.is_empty() {
        return None;
    }

    serde_json::from_str::<Value>(json_text).ok()
}

fn emit_google_text_deltas(
    chunk: &Value,
    on_text_delta: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<()> {
    let Some(candidates) = chunk.get("candidates").and_then(Value::as_array) else {
        return Ok(());
    };
    for candidate in candidates {
        let Some(content) = candidate.get("content") else {
            continue;
        };
        let Some(parts) = content.get("parts").and_then(Value::as_array) else {
            continue;
        };
        for part in parts {
            if let Some(Part::Text { text }) = parse_google_part(part) {
                on_text_delta(&text)?;
            }
        }
    }
    Ok(())
}

fn parse_google_response(body: &str) -> Result<ProviderResponse> {
    let chunks = parse_google_chunks(body)?;
    provider_response_from_google_chunks(chunks, body)
}

fn provider_response_from_google_chunks(
    chunks: Vec<Value>,
    raw_body: &str,
) -> Result<ProviderResponse> {
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
        bail!("Google response had no text or function-call parts: {raw_body}");
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

fn bridge_request(request: &ProviderRequest) -> Result<IosBridgeRequest> {
    let mut sections = Vec::new();
    if let Some(instructions) = &request.instructions {
        sections.push(format!("system: {instructions}"));
    }
    if !request.tools.is_empty() {
        sections.push(format!(
            "system: You can request robot tools by responding with one compact JSON object and no Markdown. \
When you need a tool, respond exactly as {{\"tool_calls\":[{{\"name\":\"tool_name\",\"args\":{{}}}}]}}. \
When a tool result is already present and no more tool is needed, respond with normal user-facing text. \
Available tools: {}",
            ios_bridge_tool_catalog(&request.tools)?
        ));
    }
    sections.extend(request.messages.iter().map(message_to_prompt_text));
    sections.push("assistant:".to_string());
    sections.insert(0, "system: Do not use emojis.".to_string());

    let mut media = Vec::new();
    for message in &request.messages {
        for part in &message.parts {
            if let Part::InlineData {
                mime_type,
                data,
                display_name,
            } = part
            {
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .with_context(|| {
                        format!(
                            "failed to decode inline {}{} for iPhone bridge",
                            mime_type,
                            display_name
                                .as_deref()
                                .map(|name| format!(" ({name})"))
                                .unwrap_or_default()
                        )
                    })?;
                media.push(IosBridgeMedia {
                    mime_type: mime_type.clone(),
                    bytes,
                    display_name: display_name.clone(),
                });
            }
        }
    }

    Ok(IosBridgeRequest {
        prompt: sections.join("\n\n"),
        media,
    })
}

fn ios_bridge_tool_catalog(tools: &[ToolDefinition]) -> Result<String> {
    let value = tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&value).context("failed to serialize iOS bridge tool catalog")
}

fn ios_bridge_parts_from_text(
    text: &str,
    tools: &[ToolDefinition],
    on_text_delta: &mut dyn FnMut(&str) -> Result<()>,
    already_streamed: bool,
) -> Result<Vec<Part>> {
    let trimmed = strip_json_markdown(text.trim());
    if let Some(calls) = parse_ios_bridge_tool_calls(trimmed, tools) {
        return Ok(calls
            .into_iter()
            .map(|call| Part::FunctionCall {
                name: call.name,
                args: call.args,
                thought_signature: None,
            })
            .collect());
    }
    if !text.is_empty() && !already_streamed {
        on_text_delta(text)?;
    }
    Ok(vec![Part::Text {
        text: text.to_string(),
    }])
}

fn strip_json_markdown(text: &str) -> &str {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix("```json") {
        return rest.trim().trim_end_matches("```").trim();
    }
    if let Some(rest) = trimmed.strip_prefix("```") {
        return rest.trim().trim_end_matches("```").trim();
    }
    trimmed
}

fn parse_ios_bridge_tool_calls(text: &str, tools: &[ToolDefinition]) -> Option<Vec<FunctionCall>> {
    let allowed = tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect::<Vec<_>>();

    let value: Value = match serde_json::from_str(text) {
        Ok(value) => value,
        Err(_) => return parse_truncated_ios_bridge_tool_call(text, &allowed),
    };

    let calls_value = value
        .get("tool_calls")
        .or_else(|| value.get("function_calls"))
        .or_else(|| value.get("calls"));
    if let Some(calls) = calls_value.and_then(Value::as_array) {
        let parsed = calls
            .iter()
            .filter_map(|call| parse_ios_bridge_tool_call(call, &allowed))
            .collect::<Vec<_>>();
        return (!parsed.is_empty()).then_some(parsed);
    }

    if let Some(call) = value
        .get("tool_call")
        .or_else(|| value.get("function_call"))
    {
        return parse_ios_bridge_tool_call(call, &allowed).map(|call| vec![call]);
    }

    parse_ios_bridge_tool_call(&value, &allowed).map(|call| vec![call])
}

fn parse_truncated_ios_bridge_tool_call(text: &str, allowed: &[&str]) -> Option<Vec<FunctionCall>> {
    let trimmed = text.trim();
    if !trimmed.starts_with('{')
        && !trimmed.starts_with('[')
        && !trimmed.contains("tool_call")
        && !trimmed.contains("function_call")
    {
        return None;
    }

    for name in allowed {
        if trimmed.contains(&format!("\"{name}\"")) || trimmed.contains(&format!("'{name}'")) {
            return Some(vec![FunctionCall {
                name: (*name).to_string(),
                args: json!({}),
            }]);
        }
    }
    None
}

fn parse_ios_bridge_tool_call(value: &Value, allowed: &[&str]) -> Option<FunctionCall> {
    let name = value
        .get("name")
        .or_else(|| value.get("tool"))
        .or_else(|| value.get("function"))
        .and_then(Value::as_str)?;
    if !allowed.iter().any(|allowed_name| *allowed_name == name) {
        return None;
    }
    let args = value
        .get("args")
        .or_else(|| value.get("arguments"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    Some(FunctionCall {
        name: name.to_string(),
        args,
    })
}

fn pack_ios_generate_binary(
    prompt: &str,
    max_tokens: u32,
    timeout_seconds: u64,
    media: &[IosBridgeMedia],
) -> Result<Vec<u8>> {
    let mut payload = Vec::new();
    let mut offset = 0usize;
    let mut media_headers = Vec::new();
    for item in media {
        media_headers.push(json!({
            "mime_type": item.mime_type,
            "display_name": item.display_name,
            "offset": offset,
            "bytes": item.bytes.len(),
        }));
        payload.extend_from_slice(&item.bytes);
        offset += item.bytes.len();
    }

    let header = json!({
        "type": "generate_media",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "timeout": timeout_seconds,
        "media": media_headers,
    });
    let header_bytes = serde_json::to_vec(&header)?;
    if header_bytes.len() > u32::MAX as usize {
        bail!("iPhone bridge media header is too large");
    }

    let mut frame = Vec::with_capacity(IOS_GENERATE_BINARY_MAGIC.len() + 4 + header_bytes.len() + payload.len());
    frame.extend_from_slice(IOS_GENERATE_BINARY_MAGIC);
    frame.extend_from_slice(&(header_bytes.len() as u32).to_be_bytes());
    frame.extend_from_slice(&header_bytes);
    frame.extend_from_slice(&payload);
    Ok(frame)
}

fn bridge_error_message(value: &Value) -> String {
    value
        .get("error")
        .or_else(|| value.get("message"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .unwrap_or_else(|| value.to_string())
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
    let speech_dir = expand_tilde(&args.speech_dir);
    let status_file = expand_tilde(&args.status_file);
    let recordings_dir = base_dir.join("recordings");
    fs::create_dir_all(&recordings_dir)?;
    fs::create_dir_all(&speech_dir)?;
    if let Some(parent) = status_file.parent() {
        fs::create_dir_all(parent)?;
    }

    let capture = args
        .capture_device
        .clone()
        .unwrap_or_else(|| "default".to_string());
    let mut controls = VoiceControls::new(args)?;
    let mut status = VoiceStatus::new(status_file);
    status.set_terminal(args.fullscreen_terminal);
    status.write("idle", "", &args.startup_greeting, "")?;
    controls.led(false)?;
    eprintln!(
        "Ready. Hold {} to record; release to send audio to Gemma.",
        controls.button_label()
    );

    let mut recording: Option<(Child, PathBuf, Instant, usize)> = None;
    let mut previous_pressed = false;
    let mut turn = 0usize;

    if !args.startup_agent_prompt.trim().is_empty() {
        loop {
            turn += 1;
            status.write_turn(
                turn,
                "waiting",
                "wait_for_human",
                "Waiting for a human in the camera view",
                "",
            )?;
            let human_result = agent.tools.call(&FunctionCall {
                name: "wait_for_human".to_string(),
                args: json!({
                    "stable_seconds": 0.2,
                    "timeout_seconds": 5.0,
                }),
            })?;
            if !human_result
                .get("human_detected")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                eprintln!(
                    "startup human detection did not become stable; greeting anyway: {}",
                    serde_json::to_string(&human_result).unwrap_or_else(|_| "{}".to_string())
                );
            }

            let startup_prompt = format!(
                "The startup vision gate returned this sensor result: {}. \
Greet the user now in one short spoken sentence, even if pose detection is not stable yet. \
Do not call tools and do not output JSON.",
                serde_json::to_string(&human_result).unwrap_or_else(|_| "{}".to_string())
            );
            let message_checkpoint = agent.messages.clone();
            match process_text_turn_without_tools(agent, &mut status, turn, &startup_prompt) {
                Ok(turn_run) => {
                    status.write_turn_stats(
                        turn,
                        "playing",
                        &startup_prompt,
                        &turn_run.run.text,
                        "",
                        Some(&turn_run.stats),
                    )?;
                    if let Err(error) = speak_response(args, &speech_dir, &turn_run.run.text, turn) {
                        status.write_turn_stats(
                            turn,
                            "error",
                            &startup_prompt,
                            &turn_run.run.text,
                            &error.to_string(),
                            Some(&turn_run.stats),
                        )?;
                        eprintln!("startup TTS failed: {error:?}");
                    } else {
                        status.write_turn_stats(
                            turn,
                            "idle",
                            &startup_prompt,
                            &turn_run.run.text,
                            "",
                            Some(&turn_run.stats),
                        )?;
                    }
                    break;
                }
                Err(error) => {
                    agent.messages = message_checkpoint;
                    eprintln!("startup agent turn failed: {error:?}");
                    if is_transient_startup_error(&error) {
                        status.write_turn(
                            turn,
                            "waiting",
                            "Waiting for iPhone",
                            "Waiting for iPhone Gemma runtime at ws://pi3:8765/worker",
                            "",
                        )?;
                        thread::sleep(Duration::from_secs(15));
                        continue;
                    }
                    status.write_turn(turn, "error", &startup_prompt, "", &error.to_string())?;
                    break;
                }
            }
        }
    }

    loop {
        let pressed = controls.button_pressed()?;
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
            controls.led(true)?;
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
                    controls.led(false)?;
                } else {
                    let audio_bytes = fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
                    let input = format!(
                        "Audio input: {} bytes, {:.2}s",
                        audio_bytes,
                        elapsed.as_secs_f64()
                    );
                    status.write_turn_stats(
                        turn_index,
                        "sending",
                        &input,
                        "",
                        "",
                        Some(&VoiceTurnStats {
                            audio_bytes,
                            audio_seconds: elapsed.as_secs_f64(),
                            output_tokens_estimate: 0,
                            elapsed_seconds: 0.0,
                            tokens_per_second: 0.0,
                        }),
                    )?;
                    controls.blink(Duration::from_millis(250), Duration::from_millis(250))?;
                    let result = process_audio_turn(
                        agent,
                        args,
                        &mut status,
                        turn_index,
                        &path,
                        elapsed.as_secs_f64(),
                    )
                    .with_context(|| format!("audio turn failed for {}", path.display()));
                    match result {
                        Ok(turn_run) => {
                            let input = status.last_input.clone();
                            status.write_turn_stats(
                                turn_index,
                                "playing",
                                &input,
                                &turn_run.run.text,
                                "",
                                Some(&turn_run.stats),
                            )?;
                            eprintln!(
                                "Assistant: {} tokens est, {:.1} tok/s, {} bytes audio: {}",
                                turn_run.stats.output_tokens_estimate,
                                turn_run.stats.tokens_per_second,
                                turn_run.stats.audio_bytes,
                                turn_run.run.text
                            );
                            match speak_response(args, &speech_dir, &turn_run.run.text, turn_index) {
                                Ok(()) => {
                                    status.write_turn_stats(
                                        turn_index,
                                        "idle",
                                        &input,
                                        &turn_run.run.text,
                                        "",
                                        Some(&turn_run.stats),
                                    )?;
                                }
                                Err(error) => {
                                    status.write_turn_stats(
                                        turn_index,
                                        "error",
                                        &input,
                                        &turn_run.run.text,
                                        &error.to_string(),
                                        Some(&turn_run.stats),
                                    )?;
                                    eprintln!("TTS failed: {error:?}");
                                }
                            }
                        }
                        Err(error) => {
                            status.write_turn_stats(
                                turn_index,
                                "error",
                                "Audio input",
                                "",
                                &error.to_string(),
                                Some(&VoiceTurnStats {
                                    audio_bytes,
                                    audio_seconds: elapsed.as_secs_f64(),
                                    output_tokens_estimate: 0,
                                    elapsed_seconds: 0.0,
                                    tokens_per_second: 0.0,
                                }),
                            )?;
                            eprintln!("{error:?}");
                        }
                    }
                    controls.led(false)?;
                }
            }
        }
        previous_pressed = pressed;
        thread::sleep(Duration::from_millis(25));
    }
}

fn process_audio_turn(
    agent: &mut Agent,
    args: &VoiceBotArgs,
    status: &mut VoiceStatus,
    turn: usize,
    path: &Path,
    audio_seconds: f64,
) -> Result<VoiceTurnRun> {
    let (input, parts) = voice_input_parts(args, path)?;
    let audio_bytes = fs::metadata(path)
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len();
    let started = Instant::now();
    let mut stats = VoiceTurnStats {
        audio_bytes,
        audio_seconds,
        output_tokens_estimate: 0,
        elapsed_seconds: 0.0,
        tokens_per_second: 0.0,
    };
    status.write_turn_stats(turn, "sending", &input, "", "", Some(&stats))?;
    let mut output = String::new();
    let mut stream = |delta: &str| -> Result<()> {
        output.push_str(delta);
        stats.output_tokens_estimate = estimate_tokens(&output);
        stats.elapsed_seconds = started.elapsed().as_secs_f64();
        stats.tokens_per_second = if stats.elapsed_seconds > 0.0 {
            stats.output_tokens_estimate as f64 / stats.elapsed_seconds
        } else {
            0.0
        };
        status.write_turn_stats(turn, "receiving", &input, &output, "", Some(&stats))?;
        Ok(())
    };
    let run = agent.run_with_stream(parts, &mut stream)?;
    stats.output_tokens_estimate = estimate_tokens(&run.text);
    stats.elapsed_seconds = started.elapsed().as_secs_f64();
    stats.tokens_per_second = if stats.elapsed_seconds > 0.0 {
        stats.output_tokens_estimate as f64 / stats.elapsed_seconds
    } else {
        0.0
    };
    Ok(VoiceTurnRun { run, stats })
}

fn is_transient_startup_error(error: &anyhow::Error) -> bool {
    let text = format!("{error:#}");
    text.contains("no connected iPhone worker")
        || text.contains("Gemma runtime is not ready")
        || text.contains("iOS bridge stream error")
        || text.contains("iOS bridge stream ended")
        || text.contains("failed to check iOS bridge health")
        || text.contains("Connection refused")
        || text.contains("connection refused")
}

fn process_text_turn_without_tools(
    agent: &mut Agent,
    status: &mut VoiceStatus,
    turn: usize,
    input: &str,
) -> Result<VoiceTurnRun> {
    let started = Instant::now();
    let mut stats = VoiceTurnStats {
        audio_bytes: 0,
        audio_seconds: 0.0,
        output_tokens_estimate: 0,
        elapsed_seconds: 0.0,
        tokens_per_second: 0.0,
    };
    status.write_turn_stats(turn, "thinking", input, "", "", Some(&stats))?;
    agent.messages.push(Message {
        role: "user".to_string(),
        parts: vec![Part::Text {
            text: input.to_string(),
        }],
    });

    let mut output = String::new();
    let mut stream = |delta: &str| -> Result<()> {
        output.push_str(delta);
        stats.output_tokens_estimate = estimate_tokens(&output);
        stats.elapsed_seconds = started.elapsed().as_secs_f64();
        stats.tokens_per_second = if stats.elapsed_seconds > 0.0 {
            stats.output_tokens_estimate as f64 / stats.elapsed_seconds
        } else {
            0.0
        };
        status.write_turn_stats(turn, "receiving", input, &output, "", Some(&stats))?;
        Ok(())
    };
    let request = ProviderRequest {
        model: agent.model.clone(),
        instructions: agent.instructions.clone(),
        messages: agent.messages.clone(),
        tools: Vec::new(),
        max_output_tokens: agent.max_output_tokens,
    };
    let response = agent.provider.generate(&request, &mut stream)?;
    let text = text_from_parts(&response.message.parts);
    agent.messages.push(response.message);
    stats.output_tokens_estimate = estimate_tokens(&text);
    stats.elapsed_seconds = started.elapsed().as_secs_f64();
    stats.tokens_per_second = if stats.elapsed_seconds > 0.0 {
        stats.output_tokens_estimate as f64 / stats.elapsed_seconds
    } else {
        0.0
    };
    Ok(VoiceTurnRun {
        run: AgentRun {
            text,
            messages: agent.messages.clone(),
            tool_calls: Vec::new(),
        },
        stats,
    })
}

fn voice_input_parts(args: &VoiceBotArgs, path: &Path) -> Result<(String, Vec<Part>)> {
    match effective_transcription_provider(args)? {
        TranscriptionProvider::None => {
            let bytes = fs::metadata(path)
                .with_context(|| format!("failed to stat {}", path.display()))?
                .len();
            Ok((
                format!("Audio input: {} bytes", bytes),
                vec![
                    Part::Text {
                        text: args.audio_prompt.to_string(),
                    },
                    inline_file_part(path, infer_audio_mime(path)?)?,
                ],
            ))
        }
        TranscriptionProvider::Auto => {
            unreachable!("effective_transcription_provider resolves auto")
        }
    }
}

fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count().max(if text.is_empty() { 0 } else { 1 })
}

fn effective_transcription_provider(args: &VoiceBotArgs) -> Result<TranscriptionProvider> {
    match args.transcription_provider {
        TranscriptionProvider::Auto => Ok(TranscriptionProvider::None),
        other => Ok(other),
    }
}

fn speak_response(args: &VoiceBotArgs, speech_dir: &Path, text: &str, turn: usize) -> Result<()> {
    let chunks = split_for_tts(text, args.tts_max_chars);
    if chunks.is_empty() {
        return Ok(());
    }

    if effective_tts_provider(args)? == TtsProvider::Iphone {
        for chunk in &chunks {
            iphone_tts_play(args, chunk)?;
        }
        return Ok(());
    }

    for (index, chunk) in chunks.iter().enumerate() {
        let path = speech_dir.join(format!(
            "turn-{turn:03}-tts-{index:02}.wav",
            index = index + 1
        ));
        synthesize_speech(args, chunk, &path)
            .with_context(|| format!("failed to synthesize {}", path.display()))?;
        play_wav(&path, args.playback_device.as_deref().unwrap_or("default"))
            .with_context(|| format!("failed to play {}", path.display()))?;
    }
    Ok(())
}

fn synthesize_speech(args: &VoiceBotArgs, text: &str, output: &Path) -> Result<()> {
    match effective_tts_provider(args)? {
        TtsProvider::None => Ok(()),
        TtsProvider::Command => command_tts(args, text, output),
        TtsProvider::Iphone => bail!("iPhone TTS streams directly to playback and does not synthesize a WAV file"),
        TtsProvider::Auto => unreachable!("effective_tts_provider resolves auto"),
    }
}

fn effective_tts_provider(args: &VoiceBotArgs) -> Result<TtsProvider> {
    match args.tts_provider {
        TtsProvider::Auto => {
            if args.tts_command.is_some() {
                Ok(TtsProvider::Command)
            } else if args.iphone_tts_url.is_some() || std::env::var_os("GEMMA_IOS_TTS_URL").is_some() {
                Ok(TtsProvider::Iphone)
            } else if command_exists("espeak-ng") || command_exists("espeak") {
                Ok(TtsProvider::Command)
            } else {
                bail!(
                    "no TTS provider configured; set GEMMA_IOS_TTS_URL, install espeak-ng, or pass --tts-command"
                );
            }
        }
        TtsProvider::Command if args.tts_command.is_none() => {
            if command_exists("espeak-ng") || command_exists("espeak") {
                Ok(TtsProvider::Command)
            } else {
                bail!("--tts-provider command needs --tts-command or an installed espeak/espeak-ng")
            }
        }
        other => Ok(other),
    }
}

fn iphone_tts_play(args: &VoiceBotArgs, text: &str) -> Result<()> {
    let url = args.iphone_tts_url.clone().unwrap_or_else(|| {
        let base = std::env::var("GEMMA_IOS_BRIDGE_URL")
            .unwrap_or_else(|_| DEFAULT_IOS_BRIDGE_URL.to_string());
        format!("{}/tts-stream", base.trim_end_matches('/'))
    });
    let playback_device = args.playback_device.as_deref().unwrap_or("default");
    let payload = json!({
        "text": text,
        "tts_backend": args.iphone_tts_backend,
        "voice": args.iphone_tts_voice,
        "timeout": 300,
    });
    let client = Client::builder()
        .timeout(Duration::from_secs(360))
        .build()
        .context("failed to create iPhone TTS HTTP client")?;
    let mut response = client
        .post(&url)
        .json(&payload)
        .send()
        .with_context(|| format!("iPhone TTS request failed: {url}"))?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().unwrap_or_else(|_| String::new());
        bail!("iPhone TTS error {status}: {text}");
    }

    let mut child = Command::new("aplay")
        .args([
            "-q",
            "-D",
            playback_device,
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-r",
            "24000",
            "-c",
            "1",
        ])
        .stdin(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to start aplay on {playback_device} for iPhone TTS"))?;
    let mut stdin = child.stdin.take().context("failed to open aplay stdin")?;
    let mut buffer = [0u8; 32 * 1024];
    loop {
        let read = response
            .read(&mut buffer)
            .context("failed reading iPhone TTS PCM stream")?;
        if read == 0 {
            break;
        }
        stdin
            .write_all(&buffer[..read])
            .context("failed writing iPhone TTS PCM to aplay")?;
    }
    drop(stdin);
    let status = child.wait().context("failed waiting for aplay")?;
    if !status.success() {
        bail!("aplay exited with {status}");
    }
    Ok(())
}

fn command_tts(args: &VoiceBotArgs, text: &str, output: &Path) -> Result<()> {
    if let Some(template) = &args.tts_command {
        let command = template
            .replace("{input}", text)
            .replace("{output}", &output.to_string_lossy());
        let status = Command::new("sh")
            .arg("-c")
            .arg(&command)
            .status()
            .with_context(|| format!("failed to run TTS command: {command}"))?;
        if !status.success() {
            bail!("TTS command exited with {status}");
        }
        return Ok(());
    }

    let binary = if command_exists("espeak-ng") {
        "espeak-ng"
    } else if command_exists("espeak") {
        "espeak"
    } else {
        bail!("no command TTS binary found");
    };
    let status = Command::new(binary)
        .args(["-w", output.to_string_lossy().as_ref(), text])
        .status()
        .with_context(|| format!("failed to run {binary}"))?;
    if !status.success() {
        bail!("{binary} exited with {status}");
    }
    Ok(())
}

fn play_wav(path: &Path, playback_device: &str) -> Result<()> {
    let status = Command::new("aplay")
        .args(["-q", "-D", playback_device])
        .arg(path)
        .status()
        .with_context(|| format!("failed to start aplay for {}", path.display()))?;
    if !status.success() {
        bail!("aplay exited with {status}");
    }
    Ok(())
}

fn split_for_tts(text: &str, max_chars: usize) -> Vec<String> {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return Vec::new();
    }
    if normalized.len() <= max_chars {
        return vec![normalized];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    for word in normalized.split_whitespace() {
        if !current.is_empty() && current.len() + 1 + word.len() > max_chars {
            chunks.push(current);
            current = word.to_string();
        } else if current.is_empty() {
            current = word.to_string();
        } else {
            current.push(' ');
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn command_exists(command: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {command} >/dev/null 2>&1"))
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn compact_timestamp() -> String {
    chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string()
}

struct VoiceStatus {
    path: PathBuf,
    last_input: String,
    terminal: bool,
}

impl VoiceStatus {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            last_input: String::new(),
            terminal: true,
        }
    }

    fn set_terminal(&mut self, enabled: bool) {
        self.terminal = enabled;
    }

    fn write(&mut self, state: &str, input: &str, output: &str, error: &str) -> Result<()> {
        self.write_turn_stats(0, state, input, output, error, None)
    }

    fn write_turn(
        &mut self,
        turn: usize,
        state: &str,
        input: &str,
        output: &str,
        error: &str,
    ) -> Result<()> {
        self.write_turn_stats(turn, state, input, output, error, None)
    }

    fn write_turn_stats(
        &mut self,
        turn: usize,
        state: &str,
        input: &str,
        output: &str,
        error: &str,
        stats: Option<&VoiceTurnStats>,
    ) -> Result<()> {
        self.last_input = input.to_string();
        let payload = json!({
            "mode": "voice",
            "state": state,
            "turn": turn,
            "input": input,
            "output": output,
            "error": error,
            "audio_bytes": stats.map(|stats| stats.audio_bytes),
            "audio_seconds": stats.map(|stats| stats.audio_seconds),
            "output_tokens_estimate": stats.map(|stats| stats.output_tokens_estimate),
            "elapsed_seconds": stats.map(|stats| stats.elapsed_seconds),
            "tokens_per_second": stats.map(|stats| stats.tokens_per_second),
            "updated_at": chrono::Utc::now().to_rfc3339(),
        });
        let tmp = self.path.with_extension("json.tmp");
        fs::write(&tmp, serde_json::to_string_pretty(&payload)? + "\n")?;
        fs::rename(tmp, &self.path)?;
        if self.terminal {
            render_terminal_voice(turn, state, input, output, error, stats)?;
        }
        Ok(())
    }
}

fn render_terminal_voice(
    turn: usize,
    state: &str,
    input: &str,
    output: &str,
    error: &str,
    stats: Option<&VoiceTurnStats>,
) -> Result<()> {
    print!("\x1b[2J\x1b[H\x1b[?25l");
    println!("Gemma Robot Voice");
    println!("=================");
    println!("turn: {turn}");
    println!("state: {state}");
    if let Some(stats) = stats {
        println!(
            "audio: {} bytes, {:.2}s",
            stats.audio_bytes,
            stats.audio_seconds
        );
        println!(
            "response: {} tokens est, {:.1} tok/s, {:.2}s",
            stats.output_tokens_estimate,
            stats.tokens_per_second,
            stats.elapsed_seconds
        );
    }
    println!();
    println!("input:");
    println!("{}", terminal_block(input, 8));
    println!();
    println!("response:");
    println!("{}", terminal_block(output, 14));
    if !error.is_empty() {
        println!();
        println!("error:");
        println!("{}", terminal_block(error, 6));
    }
    io::stdout().flush()?;
    Ok(())
}

fn terminal_block(text: &str, max_lines: usize) -> String {
    let mut lines: Vec<String> = text
        .lines()
        .flat_map(|line| wrap_terminal_line(line, 100))
        .collect();
    if lines.is_empty() {
        lines.push(String::new());
    }
    if lines.len() > max_lines {
        lines = lines[lines.len().saturating_sub(max_lines)..].to_vec();
    }
    lines.join("\n")
}

fn wrap_terminal_line(line: &str, width: usize) -> Vec<String> {
    if line.is_empty() {
        return vec![String::new()];
    }
    let mut out = Vec::new();
    let mut current = String::new();
    for word in line.split_whitespace() {
        if current.is_empty() {
            current.push_str(word);
        } else if current.len() + 1 + word.len() <= width {
            current.push(' ');
            current.push_str(word);
        } else {
            out.push(current);
            current = word.to_string();
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    if out.is_empty() {
        out.push(line.chars().take(width).collect());
    }
    out
}

struct VoiceControls {
    button: VoiceButton,
    led: VoiceLed,
    blink_stop: Option<Arc<AtomicBool>>,
}

impl VoiceControls {
    fn new(args: &VoiceBotArgs) -> Result<Self> {
        let sensors_file = expand_tilde(&args.sensors_file);
        let button = match args.button_source {
            ButtonSource::Gpio => VoiceButton::Gpio(GpioButton::new(args.button_gpio)?),
            ButtonSource::MicrobitSerial => VoiceButton::MicrobitSerial(
                MicrobitSerialButton::new(
                    args.microbit_device.clone(),
                    args.microbit_baud,
                    sensors_file,
                ),
            ),
        };
        let led = match args.led_source {
            LedSource::Gpio => VoiceLed::Gpio(GpioLed::new(args.led_gpio)?),
            LedSource::None => VoiceLed::None,
        };
        Ok(Self {
            button,
            led,
            blink_stop: None,
        })
    }

    fn button_label(&self) -> &str {
        self.button.label()
    }

    fn button_pressed(&mut self) -> Result<bool> {
        self.button.pressed()
    }

    fn led(&mut self, on: bool) -> Result<()> {
        self.stop_blink();
        self.led.set(on)
    }

    fn blink(&mut self, on_time: Duration, off_time: Duration) -> Result<()> {
        self.stop_blink();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        match &self.led {
            VoiceLed::Gpio(led) => {
                let led_value = led.value.clone();
                fs::write(&led_value, "1")
                    .with_context(|| format!("failed to write {}", led_value.display()))?;
                thread::spawn(move || {
                    while !worker_stop.load(Ordering::SeqCst) {
                        let _ = fs::write(&led_value, "1");
                        thread::sleep(on_time);
                        if worker_stop.load(Ordering::SeqCst) {
                            break;
                        }
                        let _ = fs::write(&led_value, "0");
                        thread::sleep(off_time);
                    }
                });
                self.blink_stop = Some(stop);
            }
            VoiceLed::None => {
                let _ = worker_stop;
            }
        }
        Ok(())
    }

    fn stop_blink(&mut self) {
        if let Some(stop) = self.blink_stop.take() {
            stop.store(true, Ordering::SeqCst);
        }
    }
}

enum VoiceButton {
    Gpio(GpioButton),
    MicrobitSerial(MicrobitSerialButton),
}

impl VoiceButton {
    fn label(&self) -> &str {
        match self {
            VoiceButton::Gpio(_) => "GPIO button",
            VoiceButton::MicrobitSerial(_) => "micro:bit A button",
        }
    }

    fn pressed(&mut self) -> Result<bool> {
        match self {
            VoiceButton::Gpio(button) => button.pressed(),
            VoiceButton::MicrobitSerial(button) => Ok(button.pressed()),
        }
    }
}

struct GpioButton {
    value: PathBuf,
}

impl GpioButton {
    fn new(pin: u32) -> Result<Self> {
        configure_gpio_input_pullup(pin);
        let number = export_gpio(pin, "in")?;
        Ok(Self {
            value: PathBuf::from(format!("/sys/class/gpio/gpio{number}/value")),
        })
    }

    fn pressed(&self) -> Result<bool> {
        let value = fs::read_to_string(&self.value)
            .with_context(|| format!("failed to read {}", self.value.display()))?;
        Ok(value.trim() == "0")
    }
}

struct MicrobitSerialButton {
    pressed: Arc<AtomicBool>,
}

impl MicrobitSerialButton {
    fn new(device: String, baud: u32, sensors_file: PathBuf) -> Self {
        let pressed = Arc::new(AtomicBool::new(false));
        let worker_pressed = pressed.clone();
        thread::spawn(move || loop {
            let Some(path) = resolve_microbit_device(&device) else {
                worker_pressed.store(false, Ordering::SeqCst);
                thread::sleep(Duration::from_secs(1));
                continue;
            };
            configure_serial_device(&path, baud);
            eprintln!("micro:bit button reader connected to {}", path.display());
            let file = match fs::File::open(&path) {
                Ok(file) => file,
                Err(error) => {
                    eprintln!("failed to open micro:bit serial {}: {error}", path.display());
                    worker_pressed.store(false, Ordering::SeqCst);
                    thread::sleep(Duration::from_secs(1));
                    continue;
                }
            };
            let mut reader = BufReader::new(file);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => {
                        worker_pressed.store(false, Ordering::SeqCst);
                        break;
                    }
                    Ok(_) => {
                        if let Some(is_pressed) = parse_microbit_button_line(&line) {
                            worker_pressed.store(is_pressed, Ordering::SeqCst);
                        }
                        if let Some(sensor_payload) = parse_microbit_sensor_line(&line) {
                            if let Err(error) = write_json_atomic(&sensors_file, &sensor_payload) {
                                eprintln!(
                                    "failed to write sensor state {}: {error:#}",
                                    sensors_file.display()
                                );
                            }
                        }
                    }
                    Err(error) => {
                        eprintln!("micro:bit serial read failed: {error}");
                        worker_pressed.store(false, Ordering::SeqCst);
                        break;
                    }
                }
            }
            thread::sleep(Duration::from_millis(250));
        });

        Self { pressed }
    }

    fn pressed(&self) -> bool {
        self.pressed.load(Ordering::SeqCst)
    }
}

enum VoiceLed {
    Gpio(GpioLed),
    None,
}

impl VoiceLed {
    fn set(&self, on: bool) -> Result<()> {
        match self {
            VoiceLed::Gpio(led) => led.set(on),
            VoiceLed::None => Ok(()),
        }
    }
}

struct GpioLed {
    value: PathBuf,
}

impl GpioLed {
    fn new(pin: u32) -> Result<Self> {
        let number = export_gpio(pin, "out")?;
        Ok(Self {
            value: PathBuf::from(format!("/sys/class/gpio/gpio{number}/value")),
        })
    }

    fn set(&self, on: bool) -> Result<()> {
        fs::write(&self.value, if on { "1" } else { "0" })
            .with_context(|| format!("failed to write {}", self.value.display()))?;
        Ok(())
    }
}

fn resolve_microbit_device(configured: &str) -> Option<PathBuf> {
    if configured != "auto" {
        let path = expand_tilde(configured);
        return path.exists().then_some(path);
    }
    let patterns = [
        "/dev/serial/by-id/*micro*",
        "/dev/serial/by-id/*Micro*",
        "/dev/serial/by-id/*CMSIS-DAP*",
        "/dev/ttyACM*",
    ];
    for pattern in patterns {
        if let Ok(output) = Command::new("sh")
            .arg("-c")
            .arg(format!("ls -1 {pattern} 2>/dev/null | head -n 1"))
            .output()
        {
            if output.status.success() {
                let text = String::from_utf8_lossy(&output.stdout);
                let path = text.trim();
                if !path.is_empty() {
                    return Some(PathBuf::from(path));
                }
            }
        }
    }
    None
}

fn configure_serial_device(path: &Path, baud: u32) {
    if command_exists("stty") {
        let _ = Command::new("stty")
            .args([
                "-F",
                path.to_string_lossy().as_ref(),
                &baud.to_string(),
                "raw",
                "-echo",
            ])
            .status();
    }
}

fn parse_microbit_button_line(line: &str) -> Option<bool> {
    let normalized = line.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }
    if normalized.contains("a:down")
        || normalized.contains("a_down")
        || normalized.contains("a=1")
        || normalized.contains("a:1")
        || normalized.contains("button_a:down")
        || normalized.contains("button_a=1")
        || normalized.contains("pressed")
    {
        return Some(true);
    }
    if normalized.contains("a:up")
        || normalized.contains("a_up")
        || normalized.contains("a=0")
        || normalized.contains("a:0")
        || normalized.contains("button_a:up")
        || normalized.contains("button_a=0")
        || normalized.contains("released")
    {
        return Some(false);
    }
    None
}

fn parse_microbit_sensor_line(line: &str) -> Option<Value> {
    let trimmed = line.trim();
    if !trimmed.contains("co2_raw=") && !trimmed.contains("co2_value=") {
        return None;
    }

    let fields = parse_microbit_key_value_line(trimmed);
    let co2_raw = fields
        .get("co2_raw")
        .and_then(|value| value.parse::<i64>().ok());
    let co2_value = fields
        .get("co2_value")
        .and_then(|value| value.parse::<i64>().ok());
    if co2_raw.is_none() && co2_value.is_none() {
        return None;
    }

    Some(json!({
        "source": "microbit",
        "sample": fields.get("sample").and_then(|value| value.parse::<u64>().ok()),
        "co2_raw": co2_raw,
        "co2_value": co2_value,
        "updated_at": chrono::Utc::now().to_rfc3339(),
        "updated_at_unix": chrono::Utc::now().timestamp(),
        "line": trimmed,
    }))
}

fn parse_microbit_key_value_line(line: &str) -> HashMap<String, String> {
    let mut fields = HashMap::new();
    for token in line.split_whitespace() {
        let Some((key, value)) = token.split_once('=') else {
            continue;
        };
        fields.insert(key.to_ascii_lowercase(), value.to_string());
    }
    fields
}

fn write_json_atomic(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_string_pretty(value)? + "\n")
        .with_context(|| format!("failed to write {}", tmp.display()))?;
    fs::rename(&tmp, path)
        .with_context(|| format!("failed to rename {} to {}", tmp.display(), path.display()))?;
    Ok(())
}

fn export_gpio(pin: u32, direction: &str) -> Result<u32> {
    let mut errors = Vec::new();
    for number in gpio_number_candidates(pin) {
        match configure_gpio(number, direction) {
            Ok(()) => return Ok(number),
            Err(error) => errors.push(format!("gpio{number}: {error:#}")),
        }
    }
    bail!("failed to export GPIO {pin}; tried {}", errors.join("; "))
}

fn configure_gpio_input_pullup(pin: u32) {
    if command_exists("pinctrl") {
        let _ = Command::new("pinctrl")
            .args(["set", &pin.to_string(), "ip", "pu"])
            .status();
    } else if command_exists("raspi-gpio") {
        let _ = Command::new("raspi-gpio")
            .args(["set", &pin.to_string(), "ip", "pu"])
            .status();
    }
}

fn configure_gpio(number: u32, direction: &str) -> Result<()> {
    let gpio_dir = PathBuf::from(format!("/sys/class/gpio/gpio{number}"));
    if !gpio_dir.exists() {
        fs::write("/sys/class/gpio/export", number.to_string())
            .with_context(|| format!("failed to export GPIO {number}"))?;
        thread::sleep(Duration::from_millis(100));
    }
    fs::write(gpio_dir.join("direction"), direction)
        .with_context(|| format!("failed to set GPIO {number} direction"))?;
    Ok(())
}

fn gpio_number_candidates(pin: u32) -> Vec<u32> {
    let mut candidates = vec![pin];
    if let Ok(entries) = fs::read_dir("/sys/class/gpio") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if !name.starts_with("gpiochip") {
                continue;
            }
            let path = entry.path();
            let base = fs::read_to_string(path.join("base"))
                .ok()
                .and_then(|text| text.trim().parse::<u32>().ok());
            let ngpio = fs::read_to_string(path.join("ngpio"))
                .ok()
                .and_then(|text| text.trim().parse::<u32>().ok());
            if let (Some(base), Some(ngpio)) = (base, ngpio) {
                if pin < ngpio {
                    candidates.push(base + pin);
                }
            }
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}
