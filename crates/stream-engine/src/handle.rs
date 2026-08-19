//! Chạy [`Session`] trên blocking thread riêng của tokio.
//!
//! Vì sao: `Session::push_pcm` chạy Silero VAD đồng bộ (~7–8 ms cho mỗi 256 ms
//! audio). Gọi thẳng từ vòng đọc WebSocket nghĩa là chiếm reactor thread — một
//! session thì không sao, nhiều session thì mọi kết nối cùng giật. Handle này đưa
//! toàn bộ phần đồng bộ sang blocking pool; phía async chỉ còn gửi chunk qua
//! channel có giới hạn (đầy thì backpressure dội về client qua TCP, không phình RAM).

use tokio::sync::mpsc;
use uuid::Uuid;

use crate::session::Session;

enum SessionInput {
    Pcm(Vec<f32>),
    /// Chốt câu đang mở (client gửi eos) nhưng session vẫn nhận audio tiếp.
    Flush,
}

pub struct SessionHandle {
    tx: mpsc::Sender<SessionInput>,
    id: Uuid,
}

impl SessionHandle {
    pub fn spawn(mut session: Session) -> Self {
        let id = session.id();
        // 64 chunk ~ vài giây audio: đủ hấp thụ jitter mạng, đủ nhỏ để không giấu
        // việc VAD tụt lại (Session còn lớp max_probe_backlog phía sau).
        let (tx, mut rx) = mpsc::channel::<SessionInput>(64);
        tokio::task::spawn_blocking(move || {
            while let Some(input) = rx.blocking_recv() {
                match input {
                    SessionInput::Pcm(pcm) => session.push_pcm(&pcm),
                    SessionInput::Flush => session.finish(),
                }
            }
            // Kênh đóng (handle bị drop / client ngắt): chốt nốt phần còn lại rồi
            // drop Session — các Sender event còn lại nằm trong task decode, chúng
            // xong thì channel event đóng và consumer tự kết thúc.
            session.finish();
        });
        Self { tx, id }
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Nạp một chunk PCM f32 16 kHz mono. Chờ khi hàng đợi đầy (backpressure).
    pub async fn push_pcm(&self, pcm: Vec<f32>) {
        if pcm.is_empty() {
            return;
        }
        if self.tx.send(SessionInput::Pcm(pcm)).await.is_err() {
            tracing::warn!(session_id = %self.id, "session task đã dừng, bỏ audio");
        }
    }

    /// Chốt câu đang mở; session vẫn sống và nhận audio tiếp.
    pub async fn flush(&self) {
        let _ = self.tx.send(SessionInput::Flush).await;
    }
}
