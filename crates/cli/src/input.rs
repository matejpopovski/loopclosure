//! Reading logs from disk and writing result files.

use flate2::read::GzDecoder;
use serde_json::Value;
use std::fs;
use std::io::Read;
use std::path::Path;

/// Read a text file, decompressing it first if it starts with the gzip magic.
pub fn read_text(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.starts_with(&[0x1f, 0x8b]) {
        let mut text = String::new();
        GzDecoder::new(&bytes[..]).read_to_string(&mut text)?;
        Ok(text)
    } else {
        Ok(String::from_utf8(bytes)?)
    }
}

pub fn write_json(path: &Path, value: &Value) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(value)? + "\n")?;
    eprintln!("wrote {}", path.display());
    Ok(())
}
