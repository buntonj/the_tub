use crate::fields;
use ndarray as nd;
use std::f64::consts::PI;

pub struct AdvectionSolver {
    pub dt: f64,
    pub vector_field: fields::VectorField2D,
    pub density: fields::ScalarField2D,
}

impl AdvectionSolver {
    pub fn step(&mut self) {
        self.advect();
    }

    pub fn step_multiple(&mut self, num_steps: u32) {
        for _ in 0..num_steps {
            self.advect();
        }
    }

    fn advect(&mut self) {
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

    pub fn evaluate_solution(&self, point: &nd::Array1<f64>) -> f64 {
        self.density.interpolate(point)
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
    let gaussian =
        move |x: f64, y: f64| gaussian_1d(x, mu[0], sigma[0]) * gaussian_1d(y, mu[1], sigma[1]);
    fields::ScalarField2D::new_from_function(axes_params, gaussian)
}

pub fn build_centered_bump(
    axes_params: [fields::AxisParams; 2],
    center: [f64; 2],
    magnitude: f64,
) -> fields::ScalarField2D {
    let bump_2d = move |x: f64, y: f64| magnitude * bump_1d(x - center[0]) * bump_1d(y - center[1]);
    fields::ScalarField2D::new_from_function(axes_params, bump_2d)
}

fn gaussian_1d(x: f64, mu: f64, sigma: f64) -> f64 {
    (-0.5 * ((x - mu) / sigma).powi(2)).exp() / (2.0 * PI).sqrt() / sigma
}

fn bump_1d(x: f64) -> f64 {
    if x.abs() < 1.0 {
        let out = (-1.0 / (1.0 - x.powi(2))).exp();
        if out.is_finite() {
            out
        } else {
            0.0
        }
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use approx::*;
    use ndarray as nd;

    use super::AdvectionSolver;
    use crate::fields;

    #[test]
    fn test_advection_operator() {
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
        let num_steps = 1;

        solver.step_multiple(1);

        // The vector field should be unchanged.
        assert_eq!(vector_field, solver.vector_field);
        // The scalar field should still be the same, but shifted left by two grid spaces.
        assert_abs_diff_eq!(
            scalar_field.field.slice(nd::s![num_steps.., ..]),
            solver.density.field.slice(nd::s![..-num_steps, ..])
        )
    }

    #[test]
    fn test_advect_zero() {
        // Test that a solver can be instantiated.
        let axes_params = [fields::AxisParams {
            start: -1.0,
            step: 0.5,
            size: 5,
        }; 2];
        let vector_field =
            fields::VectorField2D::new_from_function(axes_params, |_, _| [-1.0, 0.0]);
        let scalar_field = fields::ScalarField2D::new_zero_field(axes_params);
        let mut solver = AdvectionSolver {
            dt: axes_params[0].step,
            vector_field: vector_field.clone(),
            density: scalar_field.clone(),
        };
        let num_steps = 1;

        solver.step_multiple(num_steps);

        // The vector field should be unchanged.
        assert_eq!(vector_field, solver.vector_field);
    }

    #[test]
    fn test_mass_conservation() {
        // Test that a solver can be instantiated.
        let axes_params = [fields::AxisParams {
            start: -1.0,
            step: 0.04,
            size: 50,
        }; 2];
        let vector_field = fields::VectorField2D::new_from_function(axes_params, |_, _| [0.0, 1.0]);
        let scalar_field = fields::ScalarField2D::new_from_function(axes_params, |x, y| {
            if (x == 0.0) && (y == 0.0) {
                1.0
            } else {
                0.0
            }
        });
        let mut solver = AdvectionSolver {
            dt: 0.04,
            vector_field,
            density: scalar_field.clone(),
        };
        // Step just far enough that the mass shouldn't have "moved off the edge"
        solver.step_multiple(24);

        assert_abs_diff_eq!(
            scalar_field.field.sum(),
            solver.density.field.sum(),
            epsilon = 1E-7
        )
    }
}
