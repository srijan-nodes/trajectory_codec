use std::io::{Read, Write};
use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

pub const MAGIC_V6: &[u8; 6] = b"NAM_V6";
pub const MAGIC_V5: &[u8; 6] = b"NAM_V5";
pub const HEADER_SIZE: usize = 21;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitstreamVersion {
    V5,
    V6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    Macroblock = 0,
    LayeredTrajectory = 1,
}

#[derive(Debug, Clone)]
pub struct NamHeader {
    pub version: BitstreamVersion,
    pub height: u32,
    pub width: u32,
    pub total_frames: u32,
    pub qp: u16,
    pub stream_mode: StreamMode,
    pub has_bg: bool,
}

impl NamHeader {
    pub fn read<R: Read>(mut reader: R) -> Result<Self> {
        let mut magic = [0u8; 6];
        reader.read_exact(&mut magic).context("Failed to read magic bytes")?;

        let version = match &magic {
            MAGIC_V6 => BitstreamVersion::V6,
            MAGIC_V5 => BitstreamVersion::V5,
            other => bail!("Invalid bitstream magic: {:?}", String::from_utf8_lossy(other)),
        };

        let height = reader.read_u32::<LittleEndian>()?;
        let width = reader.read_u32::<LittleEndian>()?;
        let total_frames = reader.read_u32::<LittleEndian>()?;
        let qp = reader.read_u16::<LittleEndian>()?;

        let (stream_mode, has_bg) = match version {
            BitstreamVersion::V6 => {
                let mode_byte = reader.read_u8()?;
                let mut reserved = [0u8; 3];
                reader.read_exact(&mut reserved)?;
                let mode = match mode_byte {
                    0 => StreamMode::Macroblock,
                    1 => StreamMode::LayeredTrajectory,
                    other => bail!("Unknown stream mode byte: {}", other),
                };
                (mode, true)
            }
            BitstreamVersion::V5 => {
                let bg_byte = reader.read_u8()?;
                (StreamMode::Macroblock, bg_byte != 0)
            }
        };

        Ok(NamHeader {
            version,
            height,
            width,
            total_frames,
            qp,
            stream_mode,
            has_bg,
        })
    }

    pub fn write<W: Write>(&self, mut writer: W) -> Result<()> {
        match self.version {
            BitstreamVersion::V6 => {
                writer.write_all(MAGIC_V6)?;
                writer.write_u32::<LittleEndian>(self.height)?;
                writer.write_u32::<LittleEndian>(self.width)?;
                writer.write_u32::<LittleEndian>(self.total_frames)?;
                writer.write_u16::<LittleEndian>(self.qp)?;
                writer.write_u8(self.stream_mode as u8)?;
                writer.write_all(&[0, 0, 0])?;
            }
            BitstreamVersion::V5 => {
                writer.write_all(MAGIC_V5)?;
                writer.write_u32::<LittleEndian>(self.height)?;
                writer.write_u32::<LittleEndian>(self.width)?;
                writer.write_u32::<LittleEndian>(self.total_frames)?;
                writer.write_u16::<LittleEndian>(self.qp)?;
                writer.write_u8(if self.has_bg { 1 } else { 0 })?;
            }
        }
        Ok(())
    }
}
