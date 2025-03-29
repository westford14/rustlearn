use num::{Num, ToPrimitive};
use rustlearn_array::namedarray::NamedArray;
use rustlearn_errors::{ErrString, RustLearnError};

pub fn checks<T>(y_true: NamedArray<T>, y_pred: NamedArray<T>) -> Option<RustLearnError>
where
    T: ToPrimitive,
    T: Num,
    T: Copy,
{
    if y_true.len() != y_pred.len() {
        return Some(RustLearnError::ValidationError(ErrString::from(
            "series are not the same length",
        )));
    }
    return None;
}
