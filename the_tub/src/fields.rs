use ndarray as nd;

#[derive(Clone, Copy)]
pub struct AxisParams {
    // Parameters for building a new axis.
    pub start: f64,
    pub step: f64,
    pub size: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Axis {
    // An axis along which solutions take values.
    pub values: nd::Array1<f64>,
    // The length of the axis (counting zero as the noninclusive endpoint).
    pub length: f64,
}

#[derive(Debug)]
struct AxisLine {
    /// A line perpendicular to an axis, i.e., all values along that axis are fixed.
    // The index along the axis where this value is found.
    index: usize,
    // The fixed value along this grid line.
    value: f64,
    // The absolute distance from the query point to this axis line.
    distance: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorField2D {
    // Vector of axes arrays
    pub axes: [Axis; 2],
    // Indicates the vector field values (indexes 0 and 1 are the point coords, index 3 is the vector)
    pub field: nd::Array3<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarField2D {
    // Vector of axes arrays
    pub axes: [Axis; 2],
    // Scalar field values (indexes 0 and 1 are axes coords, value is field value)
    pub field: nd::Array2<f64>,
}

impl AxisParams {
    fn endpoint_noninclusive(&self) -> f64 {
        // compute the value of one past the axis length.
        self.start + self.step * self.size as f64
    }

    fn length(&self) -> f64 {
        // compute the length of the axis built with the structs parameters.
        self.step * (self.size as f64)
    }
}

impl Axis {
    pub fn new(axis_params: &AxisParams) -> Self {
        // Construct a new `Axis` from the provided parameters.
        Axis {
            values: nd::Array1::range(
                axis_params.start,
                axis_params.endpoint_noninclusive(),
                axis_params.step,
            ),
            length: axis_params.length(),
        }
    }

    pub fn values(&self) -> &nd::Array1<f64> {
        &self.values
    }

    pub fn array_len(&self) -> usize {
        self.values.len()
    }

    fn find_periodic_value(&self, value: f64) -> f64 {
        // Treat the axis as periodic, wrapping the given value to live in the axes limits.
        let start = self.values.first().unwrap();
        self.values.first().unwrap()
            + (value - self.values.first().unwrap()).rem_euclid(self.length)
    }

    fn find_neighbor_axis_lines(&self, value: f64) -> [AxisLine; 2] {
        // For a given value, select the two neighboring axis lines within the axis range.
        let query_value = self.find_periodic_value(value);
        let index = self
            .values
            .as_slice()
            .unwrap()
            .partition_point(|&x| x < query_value);
        let idxs = [
            index.checked_sub(1).unwrap_or(self.values.len() - 1),
            index % self.values.len(),
        ];

        // Doing it manually guarantees rem_euclid provides the right distance
        [
            AxisLine {
                index: idxs[0],
                value: self.values[idxs[0]],
                distance: (query_value - self.values[idxs[0]]).rem_euclid(self.length),
            },
            AxisLine {
                index: idxs[1],
                value: self.values[idxs[1]],
                distance: (self.values[idxs[1]] - query_value).rem_euclid(self.length),
            },
        ]
    }
}

impl VectorField2D {
    pub fn new_zero_field(axes_params: [AxisParams; 2]) -> Self {
        // Create a new square grid with zero vector field.
        VectorField2D {
            axes: [Axis::new(&axes_params[0]), Axis::new(&axes_params[1])],
            field: nd::Array3::zeros((axes_params[0].size, axes_params[1].size, 2)),
        }
    }

    pub fn assign_w_function<F: Fn(f64, f64) -> [f64; 2]>(&mut self, func: F) {
        // Populate the vector field using a closure.
        let [x_axis, y_axis] = &self.axes;
        for (i, &x) in x_axis.values().iter().enumerate() {
            for (j, &y) in y_axis.values().iter().enumerate() {
                let v = func(x, y);
                self.field[[i, j, 0]] = v[0];
                self.field[[i, j, 1]] = v[1];
            }
        }
    }

    pub fn new_from_function<F: Fn(f64, f64) -> [f64; 2]>(
        axes_params: [AxisParams; 2],
        func: F,
    ) -> Self {
        let mut field = VectorField2D::new_zero_field(axes_params);
        field.assign_w_function(func);
        field
    }

    pub fn axes(&self) -> &[Axis; 2] {
        &self.axes
    }

    pub fn num_pts(&self) -> usize {
        // How many sample points define the field.
        self.axes()
            .iter()
            .fold(1, |acc, axis| acc * axis.array_len())
    }

    pub fn num_boxes(&self) -> usize {
        self.axes()
            .iter()
            .fold(1, |acc, axis| acc * (axis.array_len() - 1))
    }

    pub fn interpolate(&self, point: &nd::Array1<f64>) -> nd::Array1<f64> {
        // Query the axes for the nearest values and distances.
        let [x_axis_lines, y_axis_lines] = [
            self.axes[0].find_neighbor_axis_lines(point[0]),
            self.axes[1].find_neighbor_axis_lines(point[1]),
        ];

        // Perform bilinear interpolation.
        // Compute denominator value.
        let denominator = (x_axis_lines[1].distance + x_axis_lines[0].distance)
            * (y_axis_lines[1].distance + y_axis_lines[0].distance);

        let mut out = nd::Array1::<f64>::zeros(2);
        // w11 term
        out += &(&self
            .field
            .slice(nd::s![x_axis_lines[0].index, y_axis_lines[0].index, ..])
            * (x_axis_lines[1].distance * y_axis_lines[1].distance));
        // w12 term
        out += &(&self
            .field
            .slice(nd::s![x_axis_lines[0].index, y_axis_lines[1].index, ..])
            * (x_axis_lines[1].distance * y_axis_lines[0].distance));
        // w21 term
        out += &(&self
            .field
            .slice(nd::s![x_axis_lines[1].index, y_axis_lines[0].index, ..])
            * (x_axis_lines[0].distance * y_axis_lines[1].distance));
        // w22 term
        out += &(&self
            .field
            .slice(nd::s![x_axis_lines[1].index, y_axis_lines[1].index, ..])
            * (x_axis_lines[0].distance * y_axis_lines[0].distance));
        out / denominator
    }
}

impl VectorField2D {
    pub fn advect_with_scalar_field(&mut self, scalar_field: &mut ScalarField2D) {
        let [x_axis, y_axis] = scalar_field.axes();
        let [x_axis_vals, y_axis_vals] = [x_axis.values(), y_axis.values()];

        let out_field =
            nd::Array2::<f64>::from_shape_fn((x_axis_vals.len(), y_axis_vals.len()), |(i, j)| {
                scalar_field.interpolate(&nd::Array1::<f64>::from_vec(vec![
                    x_axis_vals[i],
                    y_axis_vals[j],
                ]))
            });

        scalar_field.field = out_field;
    }
}

impl ScalarField2D {
    pub fn new_zero_field(axes_params: [AxisParams; 2]) -> Self {
        // Create a new scalar field filled with zeros.
        ScalarField2D {
            axes: [Axis::new(&axes_params[0]), Axis::new(&axes_params[1])],
            field: nd::Array2::zeros((axes_params[0].size, axes_params[1].size)),
        }
    }

    pub fn assign_w_function<F: Fn(f64, f64) -> f64>(&mut self, func: F) {
        // Populate the scalar field with a closure.
        let [x_axis, y_axis] = &self.axes;
        for (i, &x) in x_axis.values().iter().enumerate() {
            for (j, &y) in y_axis.values().iter().enumerate() {
                self.field[[i, j]] = func(x, y);
            }
        }
    }

    pub fn new_from_function<F: Fn(f64, f64) -> f64>(
        axes_params: [AxisParams; 2],
        func: F,
    ) -> Self {
        let mut field = ScalarField2D::new_zero_field(axes_params);
        field.assign_w_function(func);
        field
    }

    pub fn axes(&self) -> &[Axis; 2] {
        &self.axes
    }

    pub fn num_pts(&self) -> usize {
        self.axes()
            .iter()
            .fold(1, |acc, axis| acc * axis.array_len())
    }

    pub fn sum(&self) -> f64 {
        self.field.sum()
    }

    pub fn interpolate(&self, point: &nd::Array1<f64>) -> f64 {
        // Query the axes for the nearest values and distances.
        let [x_axis_lines, y_axis_lines] = [
            self.axes[0].find_neighbor_axis_lines(point[0]),
            self.axes[1].find_neighbor_axis_lines(point[1]),
        ];

        let denominator = (x_axis_lines[1].distance + x_axis_lines[0].distance)
            * (y_axis_lines[1].distance + y_axis_lines[0].distance);

        let mut out = 0.0;

        // w11 term
        out += (x_axis_lines[1].distance * y_axis_lines[1].distance)
            * &self.field[[x_axis_lines[0].index, y_axis_lines[0].index]];

        // w12 term
        out += (x_axis_lines[1].distance * y_axis_lines[0].distance)
            * &self.field[[x_axis_lines[0].index, y_axis_lines[1].index]];

        // w21 term
        out += (x_axis_lines[0].distance * y_axis_lines[1].distance)
            * &self.field[[x_axis_lines[1].index, y_axis_lines[0].index]];

        // w22 term
        out += (x_axis_lines[0].distance * y_axis_lines[0].distance)
            * &self.field[[x_axis_lines[1].index, y_axis_lines[1].index]];
        out / denominator
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use crate::fields::{AxisParams, ScalarField2D};

    use super::VectorField2D;
    use ndarray as nd;

    #[test]
    fn test_interp_vector() {
        let axes_params = [AxisParams {
            size: 4,
            start: 0.0,
            step: 0.5,
        }; 2];
        let mut vector_field = VectorField2D::new_zero_field(axes_params);
        vector_field.field.slice_mut(nd::s![0, .., ..]).fill(-1.);
        let point = nd::Array1::<f64>::from_vec(vec![0.25, 0.25]);
        let interpolated = vector_field.interpolate(&point);

        assert_eq!(interpolated[0], -0.5);
        assert_eq!(interpolated[1], -0.5);

        // Interpolating at a point by looping around should return that point again.
        let point = nd::Array1::<f64>::from_vec(vec![2.0, 2.0]);
        let interpolated = vector_field.interpolate(&point);

        assert_eq!(interpolated[0], vector_field.field[[0, 0, 0]]);
        assert_eq!(interpolated[1], vector_field.field[[0, 0, 1]]);

        // Interpolating at a target point should return that point again.
        let point = nd::Array1::<f64>::from_vec(vec![0.5, 1.5]);
        let interpolated = vector_field.interpolate(&point);

        assert_eq!(interpolated[0], vector_field.field[[1, 2, 0]]);
        assert_eq!(interpolated[1], vector_field.field[[1, 2, 1]]);
    }

    #[test]
    fn test_interp_scalar() {
        let axes_params = [AxisParams {
            size: 3,
            start: 0.0,
            step: 1.0,
        }; 2];
        let mut scalar_field = ScalarField2D::new_zero_field(axes_params);
        scalar_field.field.slice_mut(nd::s![0, ..]).fill(-1.);
        println!("{}", scalar_field.field);
        let point = nd::Array1::<f64>::from_vec(vec![0.25, 0.25]);
        let interpolated = scalar_field.interpolate(&point);

        assert_eq!(interpolated, -0.75);

        let point = nd::Array1::<f64>::from_vec(vec![0.0, 0.0]);
        let interpolated = scalar_field.interpolate(&point);
        // If we interpolate at a known point, should just get it back.
        assert_eq!(interpolated, scalar_field.field[[0, 0]]);

        // If we interpolate at a point off the grid, it should wrap.
        let point = nd::Array1::<f64>::from_vec(vec![2.25, 0.0]);
        let interpolated = scalar_field.interpolate(&point);
        println!("{}", scalar_field.field);
        assert_eq!(
            interpolated,
            0.75 * scalar_field.field[[2, 0]] + 0.25 * scalar_field.field[[0, 0]]
        );
    }

    #[test]
    fn test_populate_w_function_vector() {
        let func = |x, y| {
            if x == y {
                [1.0, 1.0]
            } else {
                [0.0, 0.0]
            }
        };
        let axes_params = [AxisParams {
            size: 2,
            start: 0.0,
            step: 1.0,
        }; 2];
        let vector_field = VectorField2D::new_from_function(axes_params, func);
        let [xs, ys] = &vector_field.axes;
        for (i, &x) in xs.values().iter().enumerate() {
            for (j, &y) in ys.values().iter().enumerate() {
                let v = func(x, y);
                assert_eq!(vector_field.field[[i, j, 0]], v[0]);
                assert_eq!(vector_field.field[[i, j, 1]], v[1]);
            }
        }
    }

    #[test]
    fn test_populate_w_function_scalar() {
        let func = |x, y| (x == y) as i32 as f64;
        let axes_params = [AxisParams {
            size: 2,
            start: 0.0,
            step: 1.0,
        }; 2];
        let scalar_field = ScalarField2D::new_from_function(axes_params, func);
        let [xs, ys] = &scalar_field.axes;
        for (i, &x) in xs.values().iter().enumerate() {
            for (j, &y) in ys.values().iter().enumerate() {
                assert_eq!(scalar_field.field[[i, j]], func(x, y));
            }
        }
    }
}
