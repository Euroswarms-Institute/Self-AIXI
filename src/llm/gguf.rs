//! GGUF v2/v3 container parsing (and a writer for synthetic test models).
//!
//! Hand-rolled on purpose — the dissection begins at the byte level. The
//! parser mmaps the file, decodes the key/value metadata table and the
//! tensor directory, and hands out bounds-checked byte slices per tensor.
//! Little-endian hosts only (x86-64 here).
//!
//! Memory policy for metadata arrays: the 248 320-entry token list is
//! retained (the bit-token probe needs it, then it can be dropped with the
//! `GgufFile`), the equally huge merges/token-type arrays are recorded as
//! `SkippedArray` — a tokenizer runtime is exactly what this crate refuses
//! to contain.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use super::quant::GgmlType;

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: usize = 32;

#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    StrArray(Vec<String>),
    IntArray(Vec<i64>),
    FloatArray(Vec<f64>),
    /// Present in the file but deliberately not materialized.
    SkippedArray {
        elem_type: u32,
        count: u64,
    },
}

impl GgufValue {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            GgufValue::U8(v) => Some(v as u64),
            GgufValue::U16(v) => Some(v as u64),
            GgufValue::U32(v) => Some(v as u64),
            GgufValue::U64(v) => Some(v),
            GgufValue::I8(v) if v >= 0 => Some(v as u64),
            GgufValue::I16(v) if v >= 0 => Some(v as u64),
            GgufValue::I32(v) if v >= 0 => Some(v as u64),
            GgufValue::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            GgufValue::F32(v) => Some(v),
            GgufValue::F64(v) => Some(v as f32),
            _ => self.as_u64().map(|v| v as f32),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// ggml dimension order: ne[0] is the contiguous row length.
    pub ne: Vec<u64>,
    pub raw_type: u32,
    pub offset: u64,
}

impl TensorInfo {
    pub fn ggml_type(&self) -> Result<GgmlType, String> {
        GgmlType::from_u32(self.raw_type).map_err(|e| format!("tensor {}: {e}", self.name))
    }

    pub fn cols(&self) -> usize {
        self.ne.first().copied().unwrap_or(1) as usize
    }

    pub fn rows(&self) -> usize {
        self.ne.iter().skip(1).product::<u64>().max(1) as usize
    }

    pub fn elems(&self) -> u64 {
        self.ne.iter().product::<u64>().max(1)
    }

    pub fn byte_len(&self) -> Result<usize, String> {
        Ok(self.ggml_type()?.row_bytes(self.cols())? * self.rows())
    }
}

struct Cur<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.b.len())
            .ok_or_else(|| format!("gguf truncated at byte {}", self.pos))?;
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        if n > 1 << 24 {
            return Err(format!("gguf string of {n} bytes is implausible"));
        }
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
}

fn scalar_width(t: u32) -> Option<usize> {
    Some(match t {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => return None,
    })
}

fn read_scalar(cur: &mut Cur, t: u32) -> Result<GgufValue, String> {
    let w = scalar_width(t).ok_or_else(|| format!("unknown gguf scalar type {t}"))?;
    let b = cur.take(w)?;
    Ok(match t {
        0 => GgufValue::U8(b[0]),
        1 => GgufValue::I8(b[0] as i8),
        2 => GgufValue::U16(u16::from_le_bytes(b.try_into().unwrap())),
        3 => GgufValue::I16(i16::from_le_bytes(b.try_into().unwrap())),
        4 => GgufValue::U32(u32::from_le_bytes(b.try_into().unwrap())),
        5 => GgufValue::I32(i32::from_le_bytes(b.try_into().unwrap())),
        6 => GgufValue::F32(f32::from_le_bytes(b.try_into().unwrap())),
        7 => GgufValue::Bool(b[0] != 0),
        10 => GgufValue::U64(u64::from_le_bytes(b.try_into().unwrap())),
        11 => GgufValue::I64(i64::from_le_bytes(b.try_into().unwrap())),
        12 => GgufValue::F64(f64::from_le_bytes(b.try_into().unwrap())),
        _ => unreachable!(),
    })
}

fn scalar_to_i64(v: &GgufValue) -> i64 {
    match *v {
        GgufValue::U8(x) => x as i64,
        GgufValue::I8(x) => x as i64,
        GgufValue::U16(x) => x as i64,
        GgufValue::I16(x) => x as i64,
        GgufValue::U32(x) => x as i64,
        GgufValue::I32(x) => x as i64,
        GgufValue::Bool(x) => x as i64,
        GgufValue::U64(x) => x as i64,
        GgufValue::I64(x) => x,
        _ => 0,
    }
}

/// Arrays kept fully materialized only where the dissection needs them.
fn retain_string_array(key: &str) -> bool {
    key == "tokenizer.ggml.tokens"
}
const MAX_RETAINED_NUMERIC_ARRAY: u64 = 4096;

pub struct GgufFile {
    mmap: Mmap,
    pub version: u32,
    pub key_order: Vec<String>,
    pub kvs: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>,
    pub alignment: usize,
    pub data_offset: usize,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        // Safety: the mapping is read-only and the file is treated as
        // immutable for the process lifetime.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
        let mut cur = Cur { b: &mmap, pos: 0 };

        if cur.take(4)? != GGUF_MAGIC {
            return Err("not a GGUF file (bad magic)".into());
        }
        let version = cur.u32()?;
        if !(2..=3).contains(&version) {
            return Err(format!("unsupported GGUF version {version}"));
        }
        let n_tensors = cur.u64()?;
        let n_kv = cur.u64()?;
        if n_tensors > 1 << 20 || n_kv > 1 << 20 {
            return Err("implausible tensor/kv count".into());
        }

        let mut key_order = Vec::with_capacity(n_kv as usize);
        let mut kvs = HashMap::with_capacity(n_kv as usize);
        for _ in 0..n_kv {
            let key = cur.string()?;
            let t = cur.u32()?;
            let value = if t == 8 {
                GgufValue::Str(cur.string()?)
            } else if t == 9 {
                let elem_type = cur.u32()?;
                let count = cur.u64()?;
                if elem_type == 8 {
                    if retain_string_array(&key) {
                        let mut v = Vec::with_capacity(count as usize);
                        for _ in 0..count {
                            v.push(cur.string()?);
                        }
                        GgufValue::StrArray(v)
                    } else {
                        for _ in 0..count {
                            let n = cur.u64()? as usize;
                            cur.take(n)?;
                        }
                        GgufValue::SkippedArray { elem_type, count }
                    }
                } else if elem_type == 9 {
                    return Err("nested gguf arrays are unsupported".into());
                } else {
                    let w = scalar_width(elem_type)
                        .ok_or_else(|| format!("unknown array elem type {elem_type}"))?;
                    if count <= MAX_RETAINED_NUMERIC_ARRAY {
                        if elem_type == 6 || elem_type == 12 {
                            let mut v = Vec::with_capacity(count as usize);
                            for _ in 0..count {
                                v.push(match read_scalar(&mut cur, elem_type)? {
                                    GgufValue::F32(x) => x as f64,
                                    GgufValue::F64(x) => x,
                                    _ => unreachable!(),
                                });
                            }
                            GgufValue::FloatArray(v)
                        } else {
                            let mut v = Vec::with_capacity(count as usize);
                            for _ in 0..count {
                                v.push(scalar_to_i64(&read_scalar(&mut cur, elem_type)?));
                            }
                            GgufValue::IntArray(v)
                        }
                    } else {
                        cur.take(w * count as usize)?;
                        GgufValue::SkippedArray { elem_type, count }
                    }
                }
            } else {
                read_scalar(&mut cur, t)?
            };
            key_order.push(key.clone());
            kvs.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(n_tensors as usize);
        let mut tensor_index = HashMap::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = cur.string()?;
            let n_dims = cur.u32()? as usize;
            if n_dims > 8 {
                return Err(format!("tensor {name}: {n_dims} dims"));
            }
            let mut ne = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                ne.push(cur.u64()?);
            }
            let raw_type = cur.u32()?;
            let offset = cur.u64()?;
            tensor_index.insert(name.clone(), tensors.len());
            tensors.push(TensorInfo {
                name,
                ne,
                raw_type,
                offset,
            });
        }

        let alignment = kvs
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT as u64) as usize;
        let data_offset = cur.pos.div_ceil(alignment) * alignment;

        Ok(GgufFile {
            mmap,
            version,
            key_order,
            kvs,
            tensors,
            tensor_index,
            alignment,
            data_offset,
        })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn tensor_data(&self, info: &TensorInfo) -> Result<&[u8], String> {
        let start = self.data_offset + info.offset as usize;
        let len = info.byte_len()?;
        self.mmap
            .get(start..start + len)
            .ok_or_else(|| format!("tensor {} data out of file bounds", info.name))
    }

    pub fn kv_str(&self, key: &str) -> Result<&str, String> {
        match self.kvs.get(key) {
            Some(GgufValue::Str(s)) => Ok(s),
            _ => Err(format!("missing string metadata {key}")),
        }
    }

    pub fn kv_u64(&self, key: &str) -> Result<u64, String> {
        self.kvs
            .get(key)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| format!("missing integer metadata {key}"))
    }

    pub fn kv_f32(&self, key: &str) -> Result<f32, String> {
        self.kvs
            .get(key)
            .and_then(|v| v.as_f32())
            .ok_or_else(|| format!("missing float metadata {key}"))
    }
}

/// Minimal GGUF v3 writer — enough to fabricate synthetic checkpoints for
/// tests and to round-trip the parser. Tensor payloads are raw block bytes.
pub fn write_gguf(
    path: &Path,
    kvs: &[(String, GgufValue)],
    tensors: &[(String, Vec<u64>, GgmlType, Vec<u8>)],
) -> Result<(), String> {
    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    let mut out = Vec::new();
    out.extend_from_slice(GGUF_MAGIC);
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());

    for (key, value) in kvs {
        put_str(&mut out, key);
        match value {
            GgufValue::U32(v) => {
                out.extend_from_slice(&4u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::I32(v) => {
                out.extend_from_slice(&5u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::F32(v) => {
                out.extend_from_slice(&6u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::Bool(v) => {
                out.extend_from_slice(&7u32.to_le_bytes());
                out.push(*v as u8);
            }
            GgufValue::Str(s) => {
                out.extend_from_slice(&8u32.to_le_bytes());
                put_str(&mut out, s);
            }
            GgufValue::U64(v) => {
                out.extend_from_slice(&10u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::StrArray(v) => {
                out.extend_from_slice(&9u32.to_le_bytes());
                out.extend_from_slice(&8u32.to_le_bytes());
                out.extend_from_slice(&(v.len() as u64).to_le_bytes());
                for s in v {
                    put_str(&mut out, s);
                }
            }
            GgufValue::IntArray(v) => {
                out.extend_from_slice(&9u32.to_le_bytes());
                out.extend_from_slice(&5u32.to_le_bytes());
                out.extend_from_slice(&(v.len() as u64).to_le_bytes());
                for x in v {
                    out.extend_from_slice(&(*x as i32).to_le_bytes());
                }
            }
            other => return Err(format!("writer does not support value {other:?}")),
        }
    }

    // Tensor directory with alignment-respecting offsets.
    let mut offset = 0u64;
    let mut offsets = Vec::with_capacity(tensors.len());
    for (name, ne, ty, data) in tensors {
        let cols = ne.first().copied().unwrap_or(1) as usize;
        let rows: usize = ne.iter().skip(1).product::<u64>().max(1) as usize;
        let expect = ty.row_bytes(cols)? * rows;
        if expect != data.len() {
            return Err(format!(
                "tensor {name}: payload {} bytes, layout wants {expect}",
                data.len()
            ));
        }
        offsets.push(offset);
        offset += data.len() as u64;
        offset = offset.div_ceil(DEFAULT_ALIGNMENT as u64) * DEFAULT_ALIGNMENT as u64;
    }
    for ((name, ne, ty, _), toff) in tensors.iter().zip(&offsets) {
        put_str(&mut out, name);
        out.extend_from_slice(&(ne.len() as u32).to_le_bytes());
        for d in ne {
            out.extend_from_slice(&d.to_le_bytes());
        }
        let raw: u32 = match ty {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q8_0 => 8,
            GgmlType::Q4K => 12,
            GgmlType::Q5K => 13,
            GgmlType::Q6K => 14,
            GgmlType::BF16 => 30,
        };
        out.extend_from_slice(&raw.to_le_bytes());
        out.extend_from_slice(&toff.to_le_bytes());
    }

    // Data section: starts at the first aligned position after the header;
    // each tensor sits at data_base + its precomputed aligned offset.
    while out.len() % DEFAULT_ALIGNMENT != 0 {
        out.push(0);
    }
    let data_base = out.len() as u64;
    for ((_, _, _, data), toff) in tensors.iter().zip(&offsets) {
        while (out.len() as u64) < data_base + toff {
            out.push(0);
        }
        out.extend_from_slice(data);
    }

    let mut f = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    f.write_all(&out).map_err(|e| format!("write: {e}"))?;
    Ok(())
}
