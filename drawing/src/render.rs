use std::f64::consts::PI;
use the_tub::fields::{Axis, ScalarField2D};

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
        vec3(0.0, 0.0, 2.0),
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        degrees(45.0),
        0.1,
        10.0,
    );

    // Let the viewer click and drag the camera.
    let mut control = OrbitControl::new(*camera.target(), 1.0, 1000.0);

    // The velocity that the triangle rotates, in radians per second.
    let mut rotation_state = RotationData {
        angle_rad: 0.0,
        rotation_frequency_hz: 0.0,
    };
    let mut stop_spinning = false;

    // let delta = 0.1;
    // let size: usize = 100;

    let physics_field = the_tub::physics::build_unit_square_gaussian_field();
    let render_axes = physics_field.axes().clone(); // Would be nice to test with alternate rendering resolution
    let renderable = ScalarField2DRenderable {
        field: physics_field,
        render_grid: render_axes,
    };

    let mut point_mesh = CpuMesh::sphere(4);
    point_mesh.transform(&Mat4::from_scale(0.01)).unwrap();
    let mut model = Gm {
        geometry: InstancedMesh::new(
            &context,
            &PointCloud {
                positions: Positions::F64(renderable.to_points()),
                colors: None,
            }
            .into(),
            &point_mesh,
        ),
        material: ColorMaterial::default(),
    };

    let mut axes = Axes::new(&context, 0.0075, 0.075);
    axes.set_transformation(Mat4::from_translation(vec3(-0.6, -0.5, 0.0)));
    let mut gui = three_d::GUI::new(&context);

    // Construct a model, with a default color material, thereby transferring the mesh data to the GPU
    //let mut model = Gm::new(Mesh::new(&context, &cpu_mesh), ColorMaterial::default());

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
                    ui.add(egui::Slider::new(&mut rotation_state.rotation_frequency_hz, -1.0..=1.0).text("Rotation Frequency (Hz)"));
                    ui.toggle_value(&mut stop_spinning, "Stop the spinning!");
                });

        });
        // Ensure the viewport matches the current window viewport which changes if the window is resized
        camera.set_viewport(frame_input.viewport);

        // Let the controls interact.
        control.handle_events(&mut camera, &mut frame_input.events);

        // Update the animation of the triangle
        if !stop_spinning {
            rotation_state.update(frame_input.elapsed_time / 1000.0);
        }
        model.set_transformation(rotation_state.current_mat4());

        // Get the screen render target to be able to render something on the screen
        frame_input.screen()
            // Clear the color and depth of the screen render target
            .clear(ClearState::color_and_depth(0.8, 0.8, 0.8, 1.0, 1.0))
            // Render the triangle with the color material which uses the per vertex colors defined at construction
            .render(
                &camera, model.into_iter().chain(&axes), &[]
            )
            .write(|| gui.render())
            .unwrap();

        // Returns default frame output to end the frame
        FrameOutput::default()
    },
    );
}

struct ScalarField2DRenderable {
    field: ScalarField2D,
    render_grid: [Axis; 2],
}

impl ScalarField2DRenderable {
    fn render(&self) -> Positions {
        Positions::F64(self.to_points())
    }
    fn to_points(&self) -> Vec<Vector3<f64>> {
        let [xaxis, yaxis] = &self.render_grid;
        let mut points = Vec::with_capacity(self.field.num_pts());
        for &x in xaxis.values() {
            for &y in yaxis.values() {
                let evaluation_point = nd::array![x, y];
                points.push(Vector3::new(x, self.field.interpolate(evaluation_point), y))
            }
        }
        points
    }
}
