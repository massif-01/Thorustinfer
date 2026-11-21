//! Stateless public protocol views over one tokenized Chat request and worker mailbox.
mod request;
mod response;
use super::*;
use axum::body::Body;
pub(crate) use request::translate;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum Kind {
    Responses,
    Messages,
    Count,
}
impl Kind {
    pub(crate) fn messages(self) -> bool {
        self != Self::Responses
    }
    pub(crate) fn path(self) -> &'static str {
        match self {
            Self::Responses => "/v1/responses",
            Self::Messages => "/v1/messages",
            Self::Count => "/v1/messages/count_tokens",
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Tool {
    name: String,
    namespace: Option<String>,
    custom: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Wire {
    pub(crate) kind: Kind,
    model: String,
    stream: bool,
    pub(super) parallel: bool,
    tools: BTreeMap<String, Tool>,
}
impl Wire {
    fn local(name: &str, namespace: Option<&str>) -> String {
        match namespace {
            None => name.to_owned(),
            Some(ns) => format!(
                "ns_{}",
                &thorustinfer_engine::artifact::sha256(&serde_json::to_vec(&(ns, name)).unwrap())
                    [..32]
            ),
        }
    }
}
pub(crate) fn headers(kind: Kind, h: &mut HeaderMap) -> Result<()> {
    if kind.messages() {
        if h.get("anthropic-version").and_then(|v| v.to_str().ok()) != Some("2023-06-01") {
            return Err("anthropic-version: expected 2023-06-01".into());
        }
        if let Some(key) = h.get("x-api-key") {
            let value = format!("Bearer {}", key.to_str().map_err(|_| "Invalid x-api-key")?);
            h.insert(
                "authorization",
                value.parse().map_err(|_| "Invalid API key header")?,
            );
        }
    }
    Ok(())
}
pub(crate) fn error_reply(kind: Kind, status: StatusCode, msg: impl Into<String>) -> Response {
    let msg = msg.into();
    if kind.messages() {
        let typ = match status.as_u16() {
            401 => "authentication_error",
            403 => "permission_error",
            404 => "not_found_error",
            413 => "request_too_large",
            429 => "rate_limit_error",
            503 => "overloaded_error",
            400..=499 => "invalid_request_error",
            _ => "api_error",
        };
        (
            status,
            Json(json!({"type":"error","error":{"type":typ,"message":msg}})),
        )
            .into_response()
    } else {
        error(status, msg)
    }
}
pub(crate) async fn error_shape(kind: Kind, r: Response) -> Response {
    if r.status().is_success() || !kind.messages() {
        return r;
    }
    let (parts, body) = r.into_parts();
    let b = axum::body::to_bytes(body, 65536).await.unwrap_or_default();
    let v = serde_json::from_slice::<Value>(&b).unwrap_or_default();
    let mut out = error_reply(
        kind,
        parts.status,
        v["error"]["message"]
            .as_str()
            .unwrap_or_else(|| std::str::from_utf8(&b).unwrap_or("Request failed")),
    );
    for key in ["retry-after", "x-request-id"] {
        if let Some(x) = parts.headers.get(key) {
            out.headers_mut().insert(key, x.clone());
        }
    }
    out
}
pub(super) fn routes(state: &Service, worker: bool) -> Router<Service> {
    let mut routes = Router::new();
    for kind in [Kind::Responses, Kind::Messages, Kind::Count] {
        let s = state.clone();
        let route = post(
            move |State(state): State<Service>,
                  axum::Extension(admission): axum::Extension<preparation::Ingress>,
                  body: std::result::Result<Json<Value>, JsonRejection>| async move {
                let body = match body {
                    Ok(Json(v)) => v,
                    Err(e) => return error_reply(kind, e.status(), e.body_text()),
                };
                let (chat, wire) = match translate(kind, body) {
                    Ok(v) => v,
                    Err(e) => return error_reply(kind, StatusCode::BAD_REQUEST, e),
                };
                prepare_chat(
                    state,
                    admission,
                    Ok(Json(chat)),
                    kind == Kind::Count,
                    Some(wire),
                )
                .await
            },
        )
        .layer(axum::middleware::from_fn(
            move |mut req: axum::extract::Request, next: axum::middleware::Next| {
                let s = s.clone();
                async move {
                    if let Err(e) = headers(kind, req.headers_mut()) {
                        return error_reply(kind, StatusCode::BAD_REQUEST, e);
                    }
                    let result = if kind == Kind::Count {
                        preparation::render_ingress(State(s), req, next).await
                    } else {
                        preparation::ingress(State(s), req, next).await
                    };
                    error_shape(kind, result).await
                }
            },
        ));
        routes = routes.route(kind.path(), route);
    }
    // Only Unix-socket workers accept already converted requests from the local manager.
    if worker {
        for (path, count) in [("/internal/wire", false), ("/internal/count", true)] {
            let route = post(
                move |State(state): State<Service>,
                      axum::Extension(admission): axum::Extension<preparation::Ingress>,
                      Json((chat, wire)): Json<(Value, Wire)>| async move {
                    prepare_chat(state, admission, Ok(Json(chat)), count, Some(wire)).await
                },
            );
            routes = routes.route(
                path,
                if count {
                    route.layer(axum::middleware::from_fn_with_state(
                        state.clone(),
                        preparation::render_ingress,
                    ))
                } else {
                    route.layer(axum::middleware::from_fn_with_state(
                        state.clone(),
                        preparation::ingress,
                    ))
                },
            );
        }
    }
    routes
}
pub(super) async fn deliver(
    w: Wire,
    id: String,
    created: u64,
    prompt: usize,
    rx: mpsc::Receiver<ModelEvent>,
) -> Response {
    let stream = w.stream;
    let kind = w.kind;
    let mut state = response::Encoder::new(w, id, created, prompt);
    if !stream {
        let mut rx = rx;
        while let Some(event) = rx.recv().await {
            match event {
                ModelEvent::Chunk(chunk) => {
                    if let Err(e) = state.chunk(&chunk) {
                        return error_reply(kind, StatusCode::INTERNAL_SERVER_ERROR, e);
                    }
                }
                ModelEvent::Complete(meta) => {
                    return match state.finish(&meta) {
                        Ok((body, _)) => Json(body).into_response(),
                        Err(e) => error_reply(kind, StatusCode::INTERNAL_SERVER_ERROR, e),
                    };
                }
                ModelEvent::Deadline(d) => {
                    return error_reply(
                        kind,
                        StatusCode::REQUEST_TIMEOUT,
                        d.payload()["error"]["message"]
                            .as_str()
                            .unwrap_or("Generation deadline"),
                    );
                }
                ModelEvent::Unavailable(e) => {
                    return error_reply(kind, StatusCode::SERVICE_UNAVAILABLE, e);
                }
                ModelEvent::Failed(e) => {
                    return error_reply(kind, StatusCode::INTERNAL_SERVER_ERROR, e);
                }
            }
        }
        return error_reply(
            kind,
            StatusCode::SERVICE_UNAVAILABLE,
            "GPU worker disconnected",
        );
    }
    let opening = state.start();
    let events = futures_util::stream::unfold(
        (rx, state, false, Some(opening)),
        move |(mut rx, mut st, mut done, opening)| async move {
            if done {
                return None;
            }
            let events = if let Some(v) = opening {
                v
            } else {
                let result = match rx.recv().await {
                    Some(ModelEvent::Chunk(v)) => st.chunk(&v),
                    Some(ModelEvent::Complete(meta)) => {
                        done = true;
                        st.finish(&meta).map(|(_, e)| e)
                    }
                    Some(ModelEvent::Failed(e) | ModelEvent::Unavailable(e)) => Err(e),
                    Some(ModelEvent::Deadline(d)) => Err(d.payload().to_string()),
                    None => Err("GPU worker disconnected before completion".into()),
                };
                result.unwrap_or_else(|e| {
                    done = true;
                    vec![st.failure(e)]
                })
            };
            Some((
                futures_util::stream::iter(events.into_iter().map(|v| {
                    Ok::<_, Infallible>(
                        Event::default()
                            .event(v["type"].as_str().unwrap_or("error"))
                            .data(v.to_string()),
                    )
                })),
                (rx, st, done, None),
            ))
        },
    )
    .flatten();
    Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response()
}
pub(crate) fn manager_routes() -> Router<Arc<crate::manager::Manager>> {
    let mut r = Router::new();
    for kind in [Kind::Responses, Kind::Messages, Kind::Count] {
        r = r.route(
            kind.path(),
            post(
                move |State(manager): State<Arc<crate::manager::Manager>>,
                      mut h: HeaderMap,
                      body: Body| async move {
                    if let Err(e) = headers(kind, &mut h) {
                        return error_reply(kind, StatusCode::BAD_REQUEST, e);
                    }
                    let r = crate::manager::routes::wire_gateway(manager, h, body, kind).await;
                    error_shape(kind, r).await
                },
            ),
        );
    }
    r
}

#[cfg(test)]
mod tests;
