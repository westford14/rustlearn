use rustlearn_array::namedarray::NamedArray;

#[derive(Debug, PartialEq, Clone)]
pub struct TrainTestSplitReturn {
    pub x_train: Vec<NamedArray<f64>>,
    pub y_train: NamedArray<f64>,
    pub x_test: Vec<NamedArray<f64>>,
    pub y_test: NamedArray<f64>,
}
