use ndarray as nd;

struct AxisLine {
    /// A line perpendicular to an axis, i.e., all values along that axis are fixed.
    // The index along the axis where this value is found.
    index: usize,
    // The fixed value along this grid line.
    value: f64,
}

struct VectorField2D {
    // Vector of axes arrays
    axes: [nd::Array1<f64>; 2],
    // Indicates the vector field values (indexes 0 and 1 are the point coords, index 3 is the vector)
    field: nd::Array3<f64>,
}

impl VectorField2D {
    fn new_empty_square_grid(grid_delta: f64, grid_size: usize) -> Self {
        // Create a new square grid with zero vector field.
        // grid_delta: 
        VectorField2D{
            axes: [nd::Array1::range(0.0, grid_delta*(grid_size as f64), grid_delta), 
                   nd::Array1::range(0.0, grid_delta*(grid_size as f64), grid_delta)],
            field: nd::Array3::zeros((grid_size, grid_size, 2))
        }
    }

    fn find_neighbor_axis_lines(&self, axis_index: usize, value: f64) -> [AxisLine; 2] {
        // For a given index, select the two neighboring axis indices while maintaining the bounds.
        let axis = &self.axes[axis_index];
        let index = self.axes[axis_index].as_slice().unwrap().partition_point(|&x| x < value );
        [index.checked_sub(1).unwrap_or(axis.len()-1),
        index % self.axes[axis_index].len(),].map(|index| AxisLine{index, value: self.axes[axis_index][index]})
    }

    fn interpolate(&self, point: nd::Array1<f64>) -> nd::Array1<f64> {
        // Find the nearest point in the provided set of points and get the value stored there
        let [x_axis_lines, y_axis_lines] = [self.find_neighbor_axis_lines(0, point[0]), self.find_neighbor_axis_lines(1, point[1])];
        
        // Perform bilinear interpolation.
        // Compute denominator value.
        let denominator = ((x_axis_lines[0].value - x_axis_lines[1].value)*(y_axis_lines[0].value-y_axis_lines[1].value)).abs();

        let mut out = nd::Array1::<f64>::zeros(2);
        // There is probably a faster/nicer way to do this with .map() and .zip() tricks?
        for x_line in x_axis_lines.as_slice() {
            for y_line in y_axis_lines.as_slice() {
                // Weighting function for bilinear interpolation
                let wt = ((point[0]-x_line.value)*(point[1]-y_line.value)/denominator).abs();
                out += &(&self.field.slice(nd::s![x_line.index, y_line.index, ..]) * wt);
            }
        }
        out

    }
}

fn main() {
    let mut vector_field = VectorField2D::new_empty_square_grid(1.0, 2);
    vector_field.field.slice_mut(nd::s![0, .., ..]).fill(1.);
    let point = nd::Array1::<f64>::from_vec(vec![0.5, 0.5]);
    let interpolated = vector_field.interpolate(point);

    let field = vector_field.field;
    println!("Field: {field:?}");
    println!("Interpolated value: {interpolated:?}");
}

#[cfg(test)]
mod tests{
    use super::VectorField2D;
    use ndarray as nd;

    #[test]
    fn test_proper_interp(){
        let mut vector_field = VectorField2D::new_empty_square_grid(1.0, 2);
        vector_field.field.slice_mut(nd::s![0, .., ..]).fill(-1.);
        let point = nd::Array1::<f64>::from_vec(vec![0.25, 0.25]);
        let interpolated = vector_field.interpolate(point);

        assert_eq!(interpolated[0], -0.25);
        assert_eq!(interpolated[1], -0.25);

    }
}