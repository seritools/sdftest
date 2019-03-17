#![feature(euclidean_division)]

use minifb::{Key, Window, WindowOptions};
use na::{Isometry3, Perspective3, Point2, Point3, Real, Unit, Vector3};
use nalgebra as na;
use rayon::prelude::*;
use std::f64::consts::FRAC_PI_2;
use std::num::Wrapping;
use std::time::{Duration, Instant};

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
const WIDTH_F: f64 = WIDTH as f64;
const HEIGHT_F: f64 = HEIGHT as f64;

const LINES_PER_WORK_ITEM: usize = 72;
const MIN_RUNTIME: Duration = Duration::from_millis(250);

fn main() {
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

    let mut window = Window::new(
        "Test - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let time = Instant::now();

        buffer
            .par_chunks_mut(WIDTH * LINES_PER_WORK_ITEM)
            .enumerate()
            .for_each(|(work_item, chunk)| {
                let lines = chunk.len() / WIDTH;
                for y in 0..lines {
                    let line = y * WIDTH;
                    for x in 0..WIDTH {
                        chunk[line + x] = run_tracer(x, (work_item * lines) + y);
                    }
                }
            });

        let duration = Instant::now() - time;

        if duration < MIN_RUNTIME {
            std::thread::sleep(MIN_RUNTIME - duration);
        }

        println!("{:?}", duration);

        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window.update_with_buffer(&buffer).unwrap();
    }
}

fn run_tracer(x: usize, y: usize) -> u32 {
    let projection = Perspective3::<f64>::new(WIDTH_F / HEIGHT_F, FRAC_PI_2, 1.0, 1000.0);
    let screen_point = Point2::new(x as f64 + 0.5, y as f64 + 0.5);

    // Compute two points in clip-space.
    // "ndc" = normalized device coordinates.
    let near_ndc_point = Point3::new(
        screen_point.x / WIDTH_F - 0.5,
        screen_point.y / HEIGHT_F - 0.5,
        -1.0,
    );
    let far_ndc_point = Point3::new(
        screen_point.x / WIDTH_F - 0.5,
        screen_point.y / HEIGHT_F - 0.5,
        1.0,
    );

    // Unproject them to view-space.
    let near_view_point = projection.unproject_point(&near_ndc_point);
    let far_view_point = projection.unproject_point(&far_ndc_point);

    // Compute the view-space line parameters.
    let line_location = near_view_point;
    let line_direction = Unit::new_normalize(far_view_point - near_view_point);

    // configure camera position/orientation
    let eye = Point3::new(0., 0., -8.0);
    let target = Point3::new(6., 0., 0.);

    // get camera to view mapping/isometry
    let camera = Isometry3::look_at_rh(&eye, &target, &Vector3::y());
    let view_to_camera = camera.inverse();

    // move view ray to camera
    let ray = Ray {
        origin: view_to_camera * line_location,
        direction: view_to_camera * line_direction,
    };

    sphere_trace(ray)
}

fn sphere_trace(ray: Ray<f64>) -> u32 {
    const THRESHOLD: f64 = 1E-6;
    const MAX_DIST: f64 = 1000.0;

    let mut ray_length = 0.0;
    let mut step_count = Wrapping(0u8);
    while ray_length < MAX_DIST {
        let cur_pos = ray.origin + (ray_length * ray.direction.into_inner());

        let dist = {
            let mut sphere_pos: Point3<f64> = cur_pos;

            p_mod1(&mut sphere_pos[0], 2.0);
            let sphere = sphere(sphere_pos, 1.);
            let floor = floor_y(cur_pos, -1.0);
            union(sphere, floor)
        };

        ray_length += dist;

        if dist < 0. {
            return rgb(255, 0, 255);
        }

        if dist < THRESHOLD * ray_length {
            return grayscale(step_count.0);
        }

        step_count += Wrapping(2u8);
    }

    // TODO: cube mapping
    rgb(80, 160, 255)
}

fn rgb(r: u8, g: u8, b: u8) -> u32 {
    u32::from(r) << 16 | u32::from(g) << 8 | u32::from(b)
}

fn grayscale(n: u8) -> u32 {
    let n = u32::from(n);
    n | n << 8 | n << 16
}

#[inline]
fn sphere(p: Point3<f64>, size: f64) -> f64 {
    p.coords.norm() - size
}

#[inline]
fn floor_y(p: Point3<f64>, offset: f64) -> f64 {
    -p[1] - offset
}

#[inline]
fn union(a: f64, b: f64) -> f64 {
    a.min(b)
}

#[inline]
fn union_chamfer(a: f64, b: f64, r: f64) -> f64 {
    a.min(b).min((a - r + b) * 0.5.sqrt())
}

#[inline]
fn p_mod1(p: &mut f64, size: f64) -> f64 {
	let halfsize = size * 0.5;
	let c = ((*p + halfsize) / size).floor();
	*p = fmod(*p + halfsize, size) - halfsize;
    c
}

#[inline]
fn fmod(x: f64, y: f64) -> f64 {
    x - y * (x/y).floor()
}

#[derive(Debug)]
struct Ray<N: Real> {
    origin: Point3<N>,
    direction: Unit<Vector3<N>>,
}
