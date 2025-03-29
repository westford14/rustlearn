use std::borrow::Cow;
use std::error::Error;
use std::fmt::{self, Debug, Display, Formatter};
use std::ops::Deref;

#[derive(Debug, Clone)]
pub struct ErrString(Cow<'static, str>);

impl<T> From<T> for ErrString
where
    T: Into<Cow<'static, str>>,
{
    fn from(msg: T) -> Self {
        ErrString(msg.into())
    }
}

impl AsRef<str> for ErrString {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Deref for ErrString {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Display for ErrString {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone)]
pub enum RustLearnError {
    ValidationError(ErrString),
    NotYetImplementedError(ErrString),
}

impl Error for RustLearnError {}

impl Display for RustLearnError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use RustLearnError::*;
        match self {
            ValidationError(msg) => write!(f, "assertion failed: {msg}"),
            NotYetImplementedError(msg) => write!(f, "not yet implemented: {msg}"),
        }
    }
}
