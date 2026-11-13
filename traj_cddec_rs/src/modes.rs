pub const BLOCK_SIZE: usize = 16;
pub const BLOCK_PIXELS: usize = BLOCK_SIZE * BLOCK_SIZE; // 256

pub const MODE_IFRAME: u8 = 0;
pub const MODE_SPATIAL: u8 = 1;
pub const MODE_RAW: u8 = 2;
pub const MODE_DYNAMIC_BOX: u8 = 3;
pub const MODE_MOTION: u8 = 4;
pub const MODE_BACKGROUND: u8 = 6;
pub const MODE_SOLID: u8 = 7;
pub const MODE_DCT_LOW: u8 = 9;
pub const MODE_DCT_MID: u8 = 10;
pub const MODE_CACHED_MOTION: u8 = 11;
pub const MODE_PALETTE: u8 = 12;
pub const MODE_FADED_MOTION: u8 = 13;
pub const MODE_FAST_DITHER: u8 = 15;
pub const MODE_SKIP: u8 = 0xFF;

// Precomputed 2-bit unpacking LUT (256 entries x 4 indices)
pub const UNPACK_2BIT_LUT: [[u8; 4]; 256] = {
    let mut lut = [[0u8; 4]; 256];
    let mut b = 0usize;
    while b < 256 {
        lut[b][0] = ((b >> 6) & 3) as u8;
        lut[b][1] = ((b >> 4) & 3) as u8;
        lut[b][2] = ((b >> 2) & 3) as u8;
        lut[b][3] = (b & 3) as u8;
        b += 1;
    }
    lut
};

// Pack 256 2-bit indices (0..=3) into 64 bytes
pub fn pack_2bit_indices(indices: &[u8], packed: &mut [u8; 64]) {
    for i in 0..64 {
        let base = i * 4;
        packed[i] = ((indices[base] & 3) << 6)
            | ((indices[base + 1] & 3) << 4)
            | ((indices[base + 2] & 3) << 2)
            | (indices[base + 3] & 3);
    }
}

// Unpack 64 bytes into 256 indices
pub fn unpack_2bit_indices(packed: &[u8], unpacked: &mut [u8; 256]) {
    for (i, &b) in packed.iter().enumerate().take(64) {
        let lut_entry = UNPACK_2BIT_LUT[b as usize];
        let base = i * 4;
        unpacked[base] = lut_entry[0];
        unpacked[base + 1] = lut_entry[1];
        unpacked[base + 2] = lut_entry[2];
        unpacked[base + 3] = lut_entry[3];
    }
}

// Orthonormal 16x16 DCT matrix: C[i, j] = alpha(i) * cos(pi * (2*j + 1) * i / 32)
// alpha(0) = sqrt(1/16) = 0.25, alpha(i) = sqrt(2/16) = 0.3535533905932738
pub struct DctEngine {
    c: [[f32; 16]; 16],
    c_t: [[f32; 16]; 16],
}

impl DctEngine {
    pub fn new() -> Self {
        let mut c = [[0.0f32; 16]; 16];
        let mut c_t = [[0.0f32; 16]; 16];
        let pi = std::f64::consts::PI;
        for i in 0..16 {
            let alpha = if i == 0 {
                (1.0 / 16.0f64).sqrt()
            } else {
                (2.0 / 16.0f64).sqrt()
            };
            for j in 0..16 {
                let val = (alpha * ((pi * (2.0 * j as f64 + 1.0) * i as f64) / 32.0).cos()) as f32;
                c[i][j] = val;
                c_t[j][i] = val;
            }
        }
        Self { c, c_t }
    }

    // Forward 2D DCT: Y = C * X * C^T
    pub fn dct2(&self, input: &[[f32; 16]; 16]) -> [[f32; 16]; 16] {
        let mut temp = [[0.0f32; 16]; 16];
        for i in 0..16 {
            for j in 0..16 {
                let mut sum = 0.0f32;
                for k in 0..16 {
                    sum += self.c[i][k] * input[k][j];
                }
                temp[i][j] = sum;
            }
        }
        let mut out = [[0.0f32; 16]; 16];
        for i in 0..16 {
            for j in 0..16 {
                let mut sum = 0.0f32;
                for k in 0..16 {
                    sum += temp[i][k] * self.c_t[k][j];
                }
                out[i][j] = sum;
            }
        }
        out
    }

    // Inverse 2D DCT for Mode 9 (4x4 low frequencies zero-padded to 16x16):
    // X = C^T[:, 0..4] * Y_4x4 * C[0..4, :]
    pub fn idct_low_4x4(&self, coeffs_4x4: &[[f32; 4]; 4]) -> [[f32; 16]; 16] {
        let mut temp = [[0.0f32; 4]; 16];
        for i in 0..16 {
            for j in 0..4 {
                let mut sum = 0.0f32;
                for k in 0..4 {
                    sum += self.c_t[i][k] * coeffs_4x4[k][j];
                }
                temp[i][j] = sum;
            }
        }
        let mut out = [[0.0f32; 16]; 16];
        for i in 0..16 {
            for j in 0..16 {
                let mut sum = 0.0f32;
                for k in 0..4 {
                    sum += temp[i][k] * self.c[k][j];
                }
                out[i][j] = sum;
            }
        }
        out
    }

    // Inverse 2D DCT for Mode 10 (8x8 mid frequencies zero-padded to 16x16):
    // X = C^T[:, 0..8] * Y_8x8 * C[0..8, :]
    pub fn idct_mid_8x8(&self, coeffs_8x8: &[[f32; 8]; 8]) -> [[f32; 16]; 16] {
        let mut temp = [[0.0f32; 8]; 16];
        for i in 0..16 {
            for j in 0..8 {
                let mut sum = 0.0f32;
                for k in 0..8 {
                    sum += self.c_t[i][k] * coeffs_8x8[k][j];
                }
                temp[i][j] = sum;
            }
        }
        let mut out = [[0.0f32; 16]; 16];
        for i in 0..16 {
            for j in 0..16 {
                let mut sum = 0.0f32;
                for k in 0..8 {
                    sum += temp[i][k] * self.c[k][j];
                }
                out[i][j] = sum;
            }
        }
        out
    }
}
