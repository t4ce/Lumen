// src/lib.rs

pub mod arch;
pub mod autograd;
pub mod backend;
pub mod parallel;
pub mod precision;
#[macro_use]
pub mod module;
pub mod init;
pub mod layers;
#[cfg(feature = "model-io")]
pub mod loader;
pub mod loss;
pub mod models;
pub mod ops;
pub mod optim;
#[cfg(feature = "cli")]
pub mod tokenizer;
