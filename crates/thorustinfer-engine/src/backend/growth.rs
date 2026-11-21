//! Backend-independent token-to-byte growth geometry; capacity is not residency.
use crate::artifact::Result;

#[derive(Clone, Copy, Debug)]
pub(crate) struct TokenLayout {
    pub row_bytes: usize,
    pub tokens_per_row: usize,
}
impl TokenLayout {
    pub fn required_bytes(self, tokens: usize) -> Result<usize> {
        if self.row_bytes == 0 || self.tokens_per_row == 0 {
            return Err("Invalid token growth geometry".into());
        }
        let bytes = tokens
            .div_ceil(self.tokens_per_row)
            .checked_mul(self.row_bytes)
            .ok_or("Token growth byte extent overflow")?;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compressed_rows_include_partial_tail_and_reject_overflow() {
        let layout = TokenLayout {
            row_bytes: 512,
            tokens_per_row: 2,
        };
        assert_eq!(layout.required_bytes(0).unwrap(), 0);
        assert_eq!(layout.required_bytes(3).unwrap(), 1024);
        assert_eq!(layout.required_bytes(8).unwrap(), 2048);
        assert!(layout.required_bytes(usize::MAX).is_err());
    }
}
