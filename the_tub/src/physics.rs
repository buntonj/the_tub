use crate::fields;
use ndarray as nd;
use std::f64::consts::PI;

struct AdvectionSolver {
    pub dt: f64,
    pub vector_field: fields::VectorField2D,
    pub density: fields::ScalarField2D,
}

impl AdvectionSolver {
    pub fn advect(&mut self) {
        let [x_axis, y_axis] = self.density.axes();
        let [x_axis_vals, y_axis_vals] = [x_axis.values(), y_axis.values()];

        let out_field =
            nd::Array2::<f64>::from_shape_fn((x_axis_vals.len(), y_axis_vals.len()), |(i, j)| {
                let coordinate = nd::Array1::<f64>::from_vec(vec![x_axis_vals[i], y_axis_vals[j]]);
                let direction = self.vector_field.interpolate(&coordinate);
                self.density
                    .interpolate(&(&coordinate - self.dt * &direction))
            });

        self.density.field = out_field;
    }
}

pub fn build_unit_square_gaussian_field() -> fields::ScalarField2D {
    build_square_gaussian_field(
        [fields::AxisParams {
            start: -1.0,
            step: 0.025,
            size: 80,
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

#[cfg(test)]
mod tests {
    use approx::*;
    use ndarray as nd;

    use super::AdvectionSolver;
    use crate::fields;

    #[test]
    fn test_advection() {
        // Test that a solver can be instantiated.
        let axes_params = [fields::AxisParams {
            start: -1.0,
            step: 0.5,
            size: 5,
        }; 2];
        let vector_field =
            fields::VectorField2D::new_from_function(axes_params, |_, _| [-1.0, 0.0]);
        let scalar_field = fields::ScalarField2D::new_from_function(axes_params, |x, y| {
            if (x == 0.0) && (y == 0.0) {
                1.0
            } else {
                0.0
            }
        });
        let mut solver = AdvectionSolver {
            dt: axes_params[0].step,
            vector_field: vector_field.clone(),
            density: scalar_field.clone(),
        };

        solver.advect();
        solver.advect();

        // The vector field should be unchanged.
        assert_eq!(vector_field, solver.vector_field);
        // The scalar field should still be the same, but shifted left by two grid spaces.
        assert_abs_diff_eq!(
            scalar_field.field.slice(nd::s![2.., ..]),
            solver.density.field.slice(nd::s![..-2, ..])
        )
    }
}
