use std::{f64::consts::PI, u8};
use the_tub::fields::{Axis, AxisParams, ScalarField2D};

use ndarray::{self as nd};
use three_d::*;

struct RotationData {
    angle_rad: f64,
    rotation_frequency_hz: f64,
}

impl RotationData {
    fn update(&mut self, dt_s: f64) {
        self.angle_rad += dt_s * 2.0 * PI * self.rotation_frequency_hz;
    }

    fn current_mat4(&mut self) -> Mat4 {
        Mat4::from_angle_y(radians(self.angle_rad as f32))
    }
}

pub fn run() {
    // Create a window (a canvas on web)
    let window = Window::new(WindowSettings {
        title: "Field!".to_string(),
        max_size: Some((1280, 720)),
        ..Default::default()
    })
    .unwrap();

    // Get the graphics context from the window
    let context = window.gl();

    // Create a camera
    let mut camera = Camera::new_perspective(
        window.viewport(),
        vec3(3.0, 2.0, 3.0),
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        degrees(45.0),
        0.1,
        10.0,
    );
    let directional_light =
        DirectionalLight::new(&context, 20.0, Srgba::WHITE, &vec3(-1.0, -1.0, -1.0));
    let ambient_light = AmbientLight::new(&context, 0.0, Srgba::WHITE);

    // Let the viewer click and drag the camera.
    let mut control = OrbitControl::new(*camera.target(), 1.0, 1000.0);

    let mut pause = false;

    let physics_axes_params = AxisParams {
        start: -1.0,
        step: 0.025,
        size: 80,
    };
    let density = the_tub::physics::build_square_gaussian_field(
        [physics_axes_params; 2],
        [0.1; 2],
        [-0.5, -0.5],
    );
    let vector_field =
        the_tub::fields::VectorField2D::new_from_function([physics_axes_params; 2], |x, y| [y, -x]);
    let solver = the_tub::physics::AdvectionSolver {
        dt: 0.05,
        vector_field,
        density,
    };
    let render_axes_params = AxisParams {
        start: -1.0,
        step: 0.02,
        size: 100,
    };
    let render_axes = [
        Axis::new(&render_axes_params),
        Axis::new(&render_axes_params),
    ];

    let mut renderable = AdvectionProblemRenderable {
        solver,
        render_axes,
    };

    let mut mesh_model = Gm::new(
        Mesh::new(&context, &renderable.to_mesh()),
        PhysicalMaterial::new_opaque(
            &context,
            &CpuMaterial {
                albedo: Srgba::new(0, 102, 204, 100),
                lighting_model: LightingModel::Cook(
                    NormalDistributionFunction::TrowbridgeReitzGGX,
                    GeometryFunction::SmithSchlickGGX,
                ),
                ..Default::default()
            },
        ),
    );

    let mut axes = Axes::new(&context, 0.0075, 0.075);
    axes.set_transformation(Mat4::from_translation(vec3(-0.6, -0.5, 0.0)));
    let mut gui = three_d::GUI::new(&context);

    // Start the main render loop
    window.render_loop(
        move |mut frame_input| // Begin a new frame with an updated frame input
    {
        gui.update(&mut frame_input.events, frame_input.accumulated_time, frame_input.viewport,frame_input.device_pixel_ratio, |gui_context| {
            egui::Window::new("left")
                .title_bar(false)
                .resizable(true)
                .constrain(true)
                .show(gui_context, |ui| {
                    ui.add_space(10.);
                    ui.heading("Control Panel");
                    ui.toggle_value(&mut pause, "Stop the animation!");
                });

        });
        // Ensure the viewport matches the current window viewport which changes if the window is resized
        camera.set_viewport(frame_input.viewport);

        // Let the controls interact.
        control.handle_events(&mut camera, &mut frame_input.events);

        // Step physics.
        if !pause {
            renderable.step();
            mesh_model.geometry = Mesh::new(&context, &renderable.to_mesh());
        }

        // Get the screen render target to be able to render something on the screen
        frame_input.screen()
            // Clear the color and depth of the screen render target
            .clear(ClearState::color_and_depth(0.8, 0.8, 0.8, 1.0, 1.0))
            // Render the triangle with the color material which uses the per vertex colors defined at construction
            .render(
                &camera, mesh_model.into_iter().chain(&axes), &[&directional_light, &ambient_light]
            )
            .write(|| gui.render())
            .unwrap();

        // Returns default frame output to end the frame
        FrameOutput::default()
    },
    );
}

struct AdvectionProblemRenderable {
    solver: the_tub::physics::AdvectionSolver,
    render_axes: [Axis; 2],
}

impl AdvectionProblemRenderable {
    fn to_points(&self) -> Vec<Vector3<f64>> {
        let [xaxis, yaxis] = &self.render_axes;
        let mut points = Vec::with_capacity(xaxis.len() * yaxis.len());
        for &x in xaxis.values() {
            for &y in yaxis.values() {
                let evaluation_point = nd::array![x, y];
                points.push(Vector3::new(
                    x,
                    self.solver.evaluate_solution(&evaluation_point),
                    y,
                ))
            }
        }
        points
    }
    fn to_indices(&self) -> Vec<u32> {
        let xlen = self.render_axes[0].len();
        let ylen = self.render_axes[1].len();
        let mut indices = Vec::<u32>::with_capacity((xlen - 1) * (ylen - 1) * 3);
        let index_map = |i, j| (i * ylen + j % ylen) as u32;
        for x in 0..(xlen - 1) {
            for y in 0..(ylen - 1) {
                indices.push(index_map(x, y));
                indices.push(index_map(x, y + 1));
                indices.push(index_map(x + 1, y + 1));

                indices.push(index_map(x + 1, y + 1));
                indices.push(index_map(x + 1, y));
                indices.push(index_map(x, y));
            }
        }
        indices
    }
    fn to_color(&self) -> Vec<Srgba> {
        (0..(self.render_axes[0].len() - 1) * (self.render_axes[1].len() - 1) * 3)
            .map(|_| Srgba::new(u8::MAX, 0, 0, 100))
            .collect()
    }
    fn to_mesh(&self) -> CpuMesh {
        let mut mesh = CpuMesh {
            indices: Indices::U32(self.to_indices()),
            positions: Positions::F64(self.to_points()),
            colors: Some(self.to_color()),
            ..Default::default()
        };
        mesh.compute_normals();
        mesh.validate().unwrap();
        mesh
    }

    fn step(&mut self) {
        self.solver.step();
    }
}

struct ScalarField2DRenderable {
    field: ScalarField2D,
    render_axes: [Axis; 2],
}

impl ScalarField2DRenderable {
    fn render(&self) -> Positions {
        Positions::F64(self.to_points())
    }
    fn to_points(&self) -> Vec<Vector3<f64>> {
        let [xaxis, yaxis] = &self.render_axes;
        let mut points = Vec::with_capacity(xaxis.len() * yaxis.len());
        for &x in xaxis.values() {
            for &y in yaxis.values() {
                let evaluation_point = nd::array![x, y];
                points.push(Vector3::new(
                    x,
                    self.field.interpolate(&evaluation_point),
                    y,
                ))
            }
        }
        points
    }
    fn to_indices(&self) -> Vec<u32> {
        let xlen = self.render_axes[0].len();
        let ylen = self.render_axes[1].len();
        let mut indices = Vec::<u32>::with_capacity((xlen - 1) * (ylen - 1) * 3);
        let index_map = |i, j| (i * ylen + j % ylen) as u32;
        for x in 0..(xlen - 1) {
            for y in 0..(ylen - 1) {
                indices.push(index_map(x, y));
                indices.push(index_map(x, y + 1));
                indices.push(index_map(x + 1, y + 1));

                indices.push(index_map(x + 1, y + 1));
                indices.push(index_map(x + 1, y));
                indices.push(index_map(x, y));
            }
        }
        indices
    }
    fn to_color(&self) -> Vec<Srgba> {
        (0..(self.render_axes[0].len() - 1) * (self.render_axes[1].len() - 1) * 3)
            .map(|_| Srgba::new(u8::MAX, 0, 0, u8::MAX))
            .collect()
    }
    fn to_mesh(&self) -> CpuMesh {
        let mut mesh = CpuMesh {
            indices: Indices::U32(self.to_indices()),
            positions: Positions::F64(self.to_points()),
            colors: Some(self.to_color()),
            ..Default::default()
        };
        mesh.compute_normals();
        mesh.validate().unwrap();
        mesh
    }
}
