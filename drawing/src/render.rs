use std::f32::consts::PI;

use three_d::*;

pub fn run() {
    // Create a window (a canvas on web)
    let window = Window::new(WindowSettings {
        title: "Triangle!".to_string(),
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

    // Create a CPU-side mesh consisting of a single colored triangle
    let positions = vec![
        vec3(0.5, -0.5, 0.0),  // bottom right
        vec3(-0.5, -0.5, 0.0), // bottom left
        vec3(0.0, 0.5, 0.0),   // top
    ];

    // The velocity that the triangle rotates, in radians per second.
    let mut rotation_frequency = 0.0;

    let colors = vec![
        Srgba::RED,   // bottom right
        Srgba::GREEN, // bottom left
        Srgba::BLUE,  // top
    ];
    let cpu_mesh = CpuMesh {
        positions: Positions::F32(positions),
        colors: Some(colors),
        ..Default::default()
    };
    let mut gui = three_d::GUI::new(&context);

    // Construct a model, with a default color material, thereby transferring the mesh data to the GPU
    let mut model = Gm::new(Mesh::new(&context, &cpu_mesh), ColorMaterial::default());

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
                    ui.add(egui::Slider::new(&mut rotation_frequency, -1.0..=1.0).text("Rotation Frequency (Hz)"));
                });

        });
        // Ensure the viewport matches the current window viewport which changes if the window is resized
        camera.set_viewport(frame_input.viewport);

        // Let the controls interact.
        control.handle_events(&mut camera, &mut frame_input.events);

        // Update the animation of the triangle
        model.set_transformation(rotate(frame_input.accumulated_time as f32, rotation_frequency));

        // Get the screen render target to be able to render something on the screen
        frame_input.screen()
            // Clear the color and depth of the screen render target
            .clear(ClearState::color_and_depth(0.8, 0.8, 0.8, 1.0, 1.0))
            // Render the triangle with the color material which uses the per vertex colors defined at construction
            .render(
                &camera, &model, &[]
            )
            .write(|| gui.render())
            .unwrap();

        // Returns default frame output to end the frame
        FrameOutput::default()
    },
    );
}

fn rotate(time: f32, vel: f32) -> Mat4 {
    Mat4::from_angle_y(radians(time / 1000.0 * 2.0 * PI * vel))
}
