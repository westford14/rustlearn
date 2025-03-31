use num::{Num, ToPrimitive};
use rustlearn_errors::RustLearnError;

#[derive(Debug, PartialEq, Clone)]
pub struct NamedArray<T> {
    pub name: String,
    pub data: Vec<T>,
}

pub type Result<NamedArray> = std::result::Result<NamedArray, RustLearnError>;

impl<T> NamedArray<T>
where
    T: Num + Copy,
{
    pub fn new(name: &str, data: Vec<T>) -> Result<NamedArray<T>> {
        Ok(NamedArray {
            name: name.to_owned(),
            data,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.data.len() == 0
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn mean(&self) -> Result<f64>
    where
        T: Into<f64>,
    {
        let sum: T = self.data.iter().copied().fold(T::zero(), T::add);
        let sum_f64: f64 = sum.into();
        let len: f64 = self.data.len() as f64;
        Ok(sum_f64 / len)
    }

    pub fn dot(&self, other: NamedArray<T>) -> Result<f64>
    where
        T: ToPrimitive,
    {
        let mut total: f64 = 0.0;
        for (i, x) in self.data.clone().iter().enumerate() {
            let x_owned = x.to_f64().unwrap();
            total += x_owned * other.data[i].to_f64().unwrap();
        }

        Ok(total)
    }
}

#[cfg(test)]
mod array_tests {
    use super::*;
    use rstest::*;

    #[fixture]
    fn named_array_fixture() -> NamedArray<f64> {
        NamedArray {
            name: "y".to_string(),
            data: vec![1.0, 2.0, 3.0, 4.0],
        }
    }

    #[rstest]
    fn test_instantiation(named_array_fixture: NamedArray<f64>) {
        drop(named_array_fixture);
    }

    #[rstest]
    fn test_is_empty(named_array_fixture: NamedArray<f64>) {
        assert!(!named_array_fixture.is_empty());
    }

    #[rstest]
    fn test_len(named_array_fixture: NamedArray<f64>) {
        assert_eq!(named_array_fixture.len(), 4)
    }

    #[rstest]
    fn test_mean(named_array_fixture: NamedArray<f64>) {
        let val = named_array_fixture.mean().unwrap();
        assert_eq!(val, 2.5)
    }

    #[rstest]
    fn test_dot_product(named_array_fixture: NamedArray<f64>) {
        let ans = named_array_fixture
            .dot(named_array_fixture.clone())
            .unwrap();
        assert_eq!(ans, 30.0)
    }
}
