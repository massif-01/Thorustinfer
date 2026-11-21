//! Shared model runtime with device-owned execution backends.
//!
pub mod architecture;
pub mod artifact;
pub mod backend;
mod batch_plan;
pub mod embedding;
pub mod error;
pub mod execution;
mod loader;
pub mod model;
pub mod mtp;
pub mod operators;
pub mod prefix;
pub mod reranking;
mod runtime;
pub mod sampling;
pub mod scheduler;
pub mod vision;
mod weights;

pub mod audio;

mod encoder_rows;
