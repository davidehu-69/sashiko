use anyhow::Result;
use async_trait::async_trait;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info};

use super::{AiProvider, AiRequest, AiResponse, CacheStats, ProviderCapabilities};

pub fn fmt_thousands(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            result.push('.');
        }
        result.push(c);
    }
    result
}

pub struct CachingAiProvider {
    inner: Arc<dyn AiProvider>,
    conn: libsql::Connection,
    session_start: i64,
    hits_this: AtomicU64,
    hits_prev: AtomicU64,
    tokens_saved_this: AtomicU64,
    tokens_saved_prev: AtomicU64,
}

impl CachingAiProvider {
    pub async fn new(inner: Arc<dyn AiProvider>, cache_path: &str, ttl_days: u64) -> Result<Self> {
        let db = libsql::Builder::new_local(cache_path).build().await?;
        let conn = db.connect()?;

        let _ = conn
            .query("PRAGMA journal_mode=WAL;", ())
            .await?
            .next()
            .await;
        let _ = conn
            .query("PRAGMA busy_timeout = 5000;", ())
            .await?
            .next()
            .await;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS response_cache (
                request_hash TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                tokens_saved INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL
            );",
        )
        .await?;

        let cutoff = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
            - ttl_days as i64 * 86400;
        let result = conn
            .execute(
                "DELETE FROM response_cache WHERE created_at < ?",
                libsql::params![cutoff],
            )
            .await;
        if let Ok(reaped) = result
            && reaped > 0
        {
            info!(
                "Response cache: reaped {} expired entries (>{} days old)",
                reaped, ttl_days
            );
        }

        let session_start = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        info!("Response cache enabled ({})", cache_path);

        Ok(Self {
            inner,
            conn,
            session_start,
            hits_this: AtomicU64::new(0),
            hits_prev: AtomicU64::new(0),
            tokens_saved_this: AtomicU64::new(0),
            tokens_saved_prev: AtomicU64::new(0),
        })
    }

    fn compute_cache_key(request: &AiRequest) -> String {
        let mut val = serde_json::to_value(request).unwrap_or_default();
        // Strip nondeterministic fields
        if let serde_json::Value::Object(ref mut map) = val {
            map.remove("context_tag");
        }
        super::scrub_thought_signatures(&mut val);
        let canonical = serde_json::to_string(&val).unwrap_or_default();
        let hash = Sha256::digest(canonical.as_bytes());
        hash.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

#[async_trait]
impl AiProvider for CachingAiProvider {
    async fn generate_content(&self, request: AiRequest) -> Result<AiResponse> {
        let hash = Self::compute_cache_key(&request);
        let hash_prefix = &hash[..12];

        let mut rows = self
            .conn
            .query(
                "SELECT response_json, tokens_saved, created_at FROM response_cache WHERE request_hash = ?",
                libsql::params![hash.clone()],
            )
            .await?;

        if let Some(row) = rows.next().await? {
            let response_json: String = row.get(0)?;
            let tokens_saved: i64 = row.get(1)?;
            let created_at: i64 = row.get(2)?;
            if let Ok(mut resp) = serde_json::from_str::<AiResponse>(&response_json) {
                let has_content = resp.content.as_ref().is_some_and(|c| !c.trim().is_empty());
                let has_tool_calls = resp.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty());

                if has_content || has_tool_calls {
                    let (origin, total) = if created_at >= self.session_start {
                        self.hits_this.fetch_add(1, Ordering::Relaxed);
                        let t = self
                            .tokens_saved_this
                            .fetch_add(tokens_saved as u64, Ordering::Relaxed)
                            + tokens_saved as u64;
                        ("this session", t)
                    } else {
                        self.hits_prev.fetch_add(1, Ordering::Relaxed);
                        let t = self
                            .tokens_saved_prev
                            .fetch_add(tokens_saved as u64, Ordering::Relaxed)
                            + tokens_saved as u64;
                        ("previous session", t)
                    };
                    info!(
                        "Cache hit [{}] ({}) — {} tokens saved (total {}: {})",
                        hash_prefix,
                        origin,
                        fmt_thousands(tokens_saved as u64),
                        origin,
                        fmt_thousands(total)
                    );
                    if let Some(ref mut usage) = resp.usage {
                        usage.cached_tokens =
                            Some(usage.cached_tokens.unwrap_or(0) + usage.prompt_tokens);
                    }
                    return Ok(resp);
                } else {
                    tracing::warn!(
                        "Found empty/invalid response in cache [{}], deleting and falling back to provider",
                        hash_prefix
                    );
                    let _ = self
                        .conn
                        .execute(
                            "DELETE FROM response_cache WHERE request_hash = ?",
                            libsql::params![hash.clone()],
                        )
                        .await;
                }
            }
        }

        debug!("Cache miss [{}]", hash_prefix);

        let resp = self.inner.generate_content(request.clone()).await?;

        let has_content = resp.content.as_ref().is_some_and(|c| !c.trim().is_empty());
        let has_tool_calls = resp.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty());

        if has_content || has_tool_calls {
            let response_json = serde_json::to_string(&resp)?;
            let request_json = serde_json::to_string(&request)?;
            let caps = self.inner.get_capabilities();
            let tokens_saved = resp
                .usage
                .as_ref()
                .map(|u| u.prompt_tokens + u.completion_tokens)
                .unwrap_or(0);
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;

            let _ = self
                .conn
                .execute(
                    "INSERT OR REPLACE INTO response_cache (request_hash, provider, model, request_json, response_json, tokens_saved, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    libsql::params![
                        hash,
                        caps.model_name.clone(),
                        caps.model_name,
                        request_json,
                        response_json,
                        tokens_saved as i64,
                        now
                    ],
                )
                .await;
        } else {
            debug!("Skipping cache insertion for empty AI response [{}]", hash_prefix);
        }

        Ok(resp)
    }

    fn estimate_tokens(&self, request: &AiRequest) -> usize {
        self.inner.estimate_tokens(request)
    }

    fn get_capabilities(&self) -> ProviderCapabilities {
        self.inner.get_capabilities()
    }

    fn cache_stats(&self) -> Option<CacheStats> {
        Some(CacheStats {
            hits_this_session: self.hits_this.load(Ordering::Relaxed),
            hits_prev_session: self.hits_prev.load(Ordering::Relaxed),
            tokens_saved_this_session: self.tokens_saved_this.load(Ordering::Relaxed),
            tokens_saved_prev_session: self.tokens_saved_prev.load(Ordering::Relaxed),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{AiMessage, AiRole};
    use tempfile::tempdir;

    struct EmptyResponseProvider {
        call_count: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl AiProvider for EmptyResponseProvider {
        async fn generate_content(&self, _request: AiRequest) -> Result<AiResponse> {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);
            if count == 0 {
                // Return empty response on first call
                Ok(AiResponse {
                    content: None,
                    thought: None,
                    thought_signature: None,
                    tool_calls: None,
                    usage: None,
                    truncated: false,
                })
            } else {
                // Return non-empty response on second call
                Ok(AiResponse {
                    content: Some("success".to_string()),
                    thought: None,
                    thought_signature: None,
                    tool_calls: None,
                    usage: None,
                    truncated: false,
                })
            }
        }

        fn estimate_tokens(&self, _request: &AiRequest) -> usize {
            0
        }

        fn get_capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                model_name: "empty-mock".to_string(),
                context_window_size: 1000,
            }
        }
    }

    #[tokio::test]
    async fn test_empty_ai_response_not_cached() -> Result<()> {
        let temp_dir = tempdir()?;
        let cache_db = temp_dir.path().join("test_cache.db");
        let cache_path = cache_db.to_string_lossy().to_string();

        let inner = Arc::new(EmptyResponseProvider {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        });

        let caching_provider = CachingAiProvider::new(inner.clone(), &cache_path, 1).await?;

        let req = AiRequest {
            system: None,
            messages: vec![AiMessage {
                role: AiRole::User,
                content: Some("hello".to_string()),
                thought: None,
                thought_signature: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: None,
            temperature: None,
            response_format: None,
            context_tag: None,
        };

        // First call should return the empty response
        let resp1 = caching_provider.generate_content(req.clone()).await?;
        assert!(resp1.content.is_none());
        assert_eq!(inner.call_count.load(Ordering::SeqCst), 1);

        // Second call should miss the cache (since empty was not cached) and call inner again
        let resp2 = caching_provider.generate_content(req.clone()).await?;
        assert_eq!(resp2.content.as_deref(), Some("success"));
        assert_eq!(inner.call_count.load(Ordering::SeqCst), 2);

        Ok(())
    }
}

