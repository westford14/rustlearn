use std::collections::HashMap;

#[derive(Debug, PartialEq)]
pub struct LinearRegressionReturn {
    pub intercept: f64,
    pub beta_values: HashMap<String, f64>,
}
