use std::f64::consts::PI;

use crate::fields;

pub fn build_unit_square_gaussian_field() -> fields::ScalarField2D {
    build_square_gaussian_field(
        [fields::AxisParams {
            start: -1.0,
            step: 0.025,
            size: 81,
        }; 2],
        [0.125; 2],
        [0.0; 2],
    )
}

pub fn build_square_gaussian_field(
    axes_params: [fields::AxisParams; 2],
    sigma: [f64; 2],
    mu: [f64; 2],
) -> fields::ScalarField2D {
    let gaussian = move |x: f64, y: f64| {
        1.0 / 2.0 / PI / (sigma[0] * sigma[1]).sqrt()
            * (-0.5 * ((x - mu[0]).powi(2) / sigma[0] + (y - mu[1]).powi(2) / sigma[1])).exp()
    };
    fields::ScalarField2D::new_from_function(axes_params, gaussian)
}
