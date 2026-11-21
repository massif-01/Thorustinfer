//! Shared model runtime with device-owned execution backends.
//!
//! CUDA executes Orin model packages. Ascend provides CANN discovery and an
//! experimental native DSv4.1 provider through the shared request interface.

pub mod architecture;
pub mod artifact;
pub mod backend;
mod batch_plan;
pub mod embedding;
pub mod error;
pub mod execution;
pub mod host_rows;
mod loader;
pub mod model;
pub mod models;
pub mod mtp;
pub mod operators;
pub mod prefix;
pub mod reranking;
mod runtime;
pub mod sampling;
pub mod scheduler;
pub mod vision;
mod weights;

/// Authoritative benchmark plan, embedded without duplicating its thresholds.
pub const BENCHMARK_PLAN: &str = include_str!("../../../configs/benchmark.json");

pub mod audio;

mod encoder_rows;
