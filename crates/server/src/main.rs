use std::net::SocketAddr;

use axum::routing::{get, post};
use axum::Router;
use tracing_subscriber::EnvFilter;

mod config;
mod http;
mod state;
mod ws;

use config::ServerConfig;
use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        // whisper.cpp log mỗi lượt VAD/inference ở mức info — quá ồn cho production.
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,whisper_rs=warn")),
        )
        .json()
        .init();
    whisper_core::install_logging_hooks();

    let cfg = ServerConfig::load()?;
    let bind_addr: SocketAddr = cfg.bind_addr.parse()?;
    tracing::info!(
        whisper_cpp = whisper_core::whisper_cpp_version(),
        model = %cfg.model.path.display(),
        max_concurrent_inference = cfg.max_concurrent_inference,
        "starting whisper-rt"
    );

    let state = AppState::init(cfg)?;
    let app = Router::new()
        .route("/health", get(http::health))
        .route("/v1/transcribe", post(http::transcribe))
        .route("/v1/stream", get(ws::stream_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    tracing::info!(%bind_addr, "server listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        tracing::error!(%err, "failed to listen for ctrl-c");
    }
    tracing::info!("shutdown signal received");
}
