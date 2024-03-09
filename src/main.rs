use bevy::prelude::*;
use bevy::input::mouse::MouseMotion;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::window::CursorGrabMode;
use bevy::render::mesh::Indices;
use bevy_rapier3d::prelude::*;
use std::thread::{self, JoinHandle};
//use bevy_scene_hook::{SceneHook,HookedSceneBundle};
use noise::{
    NoiseFn,
    Perlin,
    Seedable,
    OpenSimplex,
    Worley,
    SuperSimplex,
    Fbm,
    Min,
};
use height_mesh::ndshape::{ConstShape,ConstShape2u32};
use height_mesh::{HeightMeshBuffer,height_mesh};
use serde::{Serialize, Deserialize};
use ron::ser::{to_string_pretty, PrettyConfig};
use ron::de::from_reader;


#[derive(Serialize, Deserialize, Clone, Resource)]
pub struct Config {
    #[serde(default = "defaults::key_forward")]
    pub key_forward: KeyCode,
    #[serde(default = "defaults::key_backward")]
    pub key_backward: KeyCode,
    #[serde(default = "defaults::key_left")]
    pub key_left: KeyCode,    
    #[serde(default = "defaults::key_right")]
    pub key_right: KeyCode,
    #[serde(default = "defaults::key_jump")]
    pub key_jump: KeyCode,
    #[serde(default = "defaults::mouse_sens")]
    pub mouse_sens: f32,
    #[serde(default = "defaults::terrain_height")]
    pub terrain_height: f32,
    #[serde(default = "defaults::terrain_scale")]
    pub terrain_scale: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            key_forward: KeyCode::W,
            key_backward: KeyCode::S,
            key_left: KeyCode::A,
            key_right: KeyCode::D,
            key_jump: KeyCode::Space,
            mouse_sens: 1.0,
            terrain_height: 13.,
            terrain_scale: 0.02,
        }
    }
}


macro_rules! default_ {
    ($name:ident, $type:ident) => {
        pub fn $name() -> $type {
            Config::default().$name
        }
    };
}

mod defaults {
    use super::Config;
    use bevy::prelude::KeyCode;
    default_!(key_forward, KeyCode);
    default_!(key_backward, KeyCode);
    default_!(key_left, KeyCode);
    default_!(key_right, KeyCode);
    default_!(key_jump, KeyCode);
    default_!(mouse_sens, f32);
    default_!(terrain_height, f32);
    default_!(terrain_scale, f32);
}

impl Config {
    pub fn load() -> Self {
        let res = std::env::current_exe();
        let exe_path = match res {
            Ok(config_dir) => config_dir,
            Err(_) => return Self::default(),
        };
        let config_dir = match exe_path.parent() {
            Some(config_dir) => config_dir,
            None => return Self::default(),
        };
        //let _ = std::fs::create_dir_all(&config_dir);
        let config_path = config_dir.join("desert_surf.cfg");
        let file = std::fs::File::open(&config_path);
        let file = match file {
            Ok(file) => {file},
            Err(_) => {
                _ = std::fs::write(config_path, to_string_pretty(&Self::default(), PrettyConfig::default()).unwrap());
                return Self::default()
            }
        };
        let conf = from_reader::<std::fs::File, Self>(file);
        match conf {
            Ok(conf) => {
                // write back default values of any fields not present
                _ = std::fs::write(config_path, to_string_pretty(&conf, PrettyConfig::default()).unwrap());
                conf
            },
            Err(_) => Self::default(),
        }
    }
}

const PHYSICS_TIMESTEP: f32 = 0.015625;// 64 fps

static mut CONF_TERRAIN_HEIGHT: f32 = 13.;
static mut CONF_TERRAIN_SCALE: f32 = 0.02;

fn main() {
    let conf = Config::load();

    unsafe {
        CONF_TERRAIN_HEIGHT = conf.terrain_height;
        CONF_TERRAIN_SCALE = conf.terrain_scale;
    }

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        //.add_plugins(RapierDebugRenderPlugin::default())
        .add_plugins(bevy_scene_hook::HookPlugin)
        .add_systems(Startup, (add_player, /*add_debug_text*/))
        .add_systems(Update, (rotate_camera, move_camera, do_jump_buffering, cursor_locking, load_unload_chunks, load_unload_visual_chunks, animate_chunk_in,modify_render_distance,check_loading_chunks))
        .add_systems(FixedUpdate, move_and_slide)
        .add_systems(FixedUpdate, get_wishdir.pipe(movement).after(move_and_slide))
        .insert_resource(FixedTime::new_from_secs(PHYSICS_TIMESTEP))
        .insert_resource(SandMaterial {0: None})
        .insert_resource(ClearColor(Color::hex("C4AA82").unwrap()))
        .insert_resource(RenderDistance {0: 3})
        .run();
}

#[derive(Resource)]
struct RenderDistance(i32);

#[derive(Resource)]
struct SandMaterial(Option<Handle<StandardMaterial>>);

#[derive(Component, Default)]
struct CameraRotation {
    yaw: f32,
    pitch: f32,
}

#[derive(Component)]
struct KeybindsComponent {
    forward: KeyCode,
    backward: KeyCode,
    left: KeyCode,
    right: KeyCode,
    jump: KeyCode,
    sensitivity: f32,
}

impl Default for KeybindsComponent {
    fn default() -> Self {
        let conf = Config::load();
        Self { 
            forward: conf.key_forward, 
            backward: conf.key_backward, 
            left: conf.key_left, 
            right: conf.key_right, 
            jump: conf.key_jump, 
            sensitivity: conf.mouse_sens * 0.001, 
        }
    }
}

#[derive(Component, Default)]
struct JumpBuffer(bool);

#[derive(Component, Default)]
struct Grounded(bool);

#[derive(Component)]
struct Player {
    vel: Vec3,
}
/*
#[derive(Bundle)] no longer using rigidbody physics
struct PlayerBodyBundle {
    transform: TransformBundle,
    rb: RigidBody,
    col: Collider,
    res: Restitution,
    ccd: Ccd,
    vel: Velocity,
    lock: LockedAxes,
    grav: GravityScale,
}

impl Default for PlayerBodyBundle {
    fn default() -> Self {
        Self {
            col: Collider::capsule_y(1., 0.8), 
            transform: TransformBundle::default(), 
            rb: RigidBody::KinematicVelocityBased, 
            res: Restitution::coefficient(0.),
            ccd: Ccd::enabled(),
            vel: Velocity::default(),
            lock: LockedAxes::ROTATION_LOCKED,
            grav: GravityScale(8.)
        }
    }
}*/

#[derive(Bundle)]
struct PlayerObject {
    jump_buffer: JumpBuffer,
    grounded: Grounded,
    keybinds: KeybindsComponent,
    _p: Player,
    transform: TransformBundle,
//    collider: Collider,
}

impl Default for PlayerObject {
    fn default() -> Self {
        Self { 
            jump_buffer: JumpBuffer::default(),
            grounded: Grounded::default(),
            keybinds: KeybindsComponent::default(), 
            _p: Player{vel:Vec3::ZERO},
            transform: TransformBundle::from_transform(Transform::from_xyz(0., 200., 0.)),
//            collider: Collider::capsule_y(1., 0.8),
        }
    }
}

#[derive(Bundle)]
struct CameraBundle {
    camera_rotation: CameraRotation,
    cam: Camera3dBundle,
}

impl Default for CameraBundle {
    fn default() -> Self {
        Self {
            camera_rotation: CameraRotation::default(),
            cam: Camera3dBundle::default(),
        }
    }
}

fn add_player(mut commands: Commands, assets: Res<AssetServer>, mut sand: ResMut<SandMaterial>, mut mats: ResMut<Assets<StandardMaterial>>){
    commands.spawn(PlayerObject::default());
    commands.spawn((CameraBundle::default()/*,
    FogSettings {
            color: Color::hex("FAE467").unwrap(),
            falloff: FogFalloff::Linear { start: 48., end: 95. },
            directional_light_color: Color::NONE,
            directional_light_exponent: 0.0,
        }*/));

    sand.0 = Some(mats.add(StandardMaterial {base_color: Color::hex("F0CA92").unwrap(), metallic: 1.0, perceptual_roughness: 1.0, ..Default::default()}));

    commands.spawn(DirectionalLightBundle {directional_light: DirectionalLight { color: Color::WHITE, illuminance: 12000.0, shadows_enabled: false, shadow_depth_bias: 0.0, shadow_normal_bias: 0.0 }, transform: Transform::from_xyz(1.,1.,1.).looking_at(Vec3::ZERO, Vec3::Y), ..Default::default()});
    /*commands.spawn(HookedSceneBundle {
        scene: SceneBundle {scene: assets.load("testmap.glb#Scene0"), ..default()},
        hook: SceneHook::new(|entity, cmds|{
            let meshes = entity.world().resource::<Assets<Mesh>>();
            if entity.contains::<Handle<Mesh>>() {
                cmds.insert(Collider::from_bevy_mesh(meshes.get(entity.get::<Handle<Mesh>>().unwrap()).unwrap(), &ComputedColliderShape::TriMesh).unwrap());
            }
        })
    });  */
    
    //let heights = generate_chunk(Vec2::new(-16.,-16.));

    //let heightfield = Collider::heightfield(heights, CHUNK_TILE_LENGTH as usize, CHUNK_TILE_LENGTH as usize, Vec3::new(CHUNK_TILE_LENGTH as f32, 1., CHUNK_TILE_LENGTH as f32));

    //commands.spawn((heightfield, Transform::from_xyz(-16., -20., -16.)));
}

#[derive(Component, Default)]
struct Chunk {
    x: i32,
    y: i32,
}

fn chunk_coordinate_of(v: Vec2) -> (i32,i32) {
    let v = v + Vec2::new((CHUNK_TILE_LENGTH/2) as f32,(CHUNK_TILE_LENGTH/2) as f32);
    let scaled = v / CHUNK_TILE_LENGTH as f32;
    let floored = scaled.floor();
    (floored.x as i32, floored.y as i32)
}

fn load_unload_chunks(mut commands: Commands, player: Query<(&Player, &Transform)>, already_loaded: Query<(Entity, &Chunk)>) {
    let player = player.get_single();
    if let Ok(player) = player {
        let middle_chunk = chunk_coordinate_of(get_horizontal(player.1.translation));
        let mut near_chunks = vec!(
            middle_chunk,
            (middle_chunk.0 + 1, middle_chunk.1),
            (middle_chunk.0 + 1, middle_chunk.1 + 1),
            (middle_chunk.0, middle_chunk.1 + 1),
            (middle_chunk.0 - 1, middle_chunk.1 + 1),
            (middle_chunk.0 - 1, middle_chunk.1),
            (middle_chunk.0 - 1, middle_chunk.1 - 1),
            (middle_chunk.0, middle_chunk.1 - 1),
            (middle_chunk.0 + 1, middle_chunk.1 - 1),
        );  

        for (entity, chunk) in already_loaded.iter() {
            if !near_chunks.contains(&(chunk.x,chunk.y)) {
                commands.entity(entity).despawn()
            } else {
                let index = near_chunks.iter().position(| &pos| pos == (chunk.x,chunk.y)).unwrap();
                near_chunks.remove(index);
            }
        }

        for (x, y) in near_chunks {
            /*let heights = generate_chunk(Vec2::new(
                (x * CHUNK_TILE_LENGTH as i32) as f32 - (CHUNK_TILE_LENGTH / 2) as f32,
                (y * CHUNK_TILE_LENGTH as i32) as f32 - (CHUNK_TILE_LENGTH / 2) as f32)
            );*/

            let mut chunk = commands.spawn((Chunk {x, y},TransformBundle::from_transform(Transform::from_xyz((x*CHUNK_TILE_LENGTH as i32) as f32, 0., (y*CHUNK_TILE_LENGTH as i32) as f32))));

            //let thread_handle = thread::spawn(move || {

            //    let index = chunk.id().index();
            //    let generation = chunk.id().generation();

                let heights = generate_chunk(Vec2::new(
                    (x * CHUNK_TILE_LENGTH as i32) as f32 - (CHUNK_TILE_LENGTH / 2) as f32,
                    (y * CHUNK_TILE_LENGTH as i32) as f32 - (CHUNK_TILE_LENGTH / 2) as f32),
                    CHUNK_TILE_LENGTH,
                );

                let heightfield = Collider::heightfield(heights, CHUNK_TILE_LENGTH as usize, CHUNK_TILE_LENGTH as usize, Vec3::new(CHUNK_TILE_LENGTH as f32, 1., CHUNK_TILE_LENGTH as f32));
                chunk.insert(heightfield);
                //chunk.insert(heightfield);
            //});

            /*let heightfield = Collider::heightfield(heights, CHUNK_TILE_LENGTH as usize,
                * CHUNK_TILE_LENGTH as usize, Vec3::new(CHUNK_TILE_LENGTH as f32, 1.,
                    * CHUNK_TILE_LENGTH as f32));*/
            /*commands.spawn((heightfield, Chunk {x, y},
                TransformBundle::from_transform(Transform::from_xyz((x*CHUNK_TILE_LENGTH as i32) as f32, 0., (y*CHUNK_TILE_LENGTH as i32) as f32)))
            );*/
            //});

            //chunk.insert(LoadingCollisionChunk(Some(thread_handle)));
        }
    }
}

const VIS_CHUNK_RESOLUTION: u32 = 67;

type ChunkShape = ConstShape2u32<VIS_CHUNK_RESOLUTION,VIS_CHUNK_RESOLUTION>;

#[derive(Component, Default)]
struct VisualChunk {
    x: i32,
    y: i32,
}

#[derive(Component)]
struct VisualChunkAnimating;

fn modify_render_distance(mut render_distance: ResMut<RenderDistance>, keys: Res<Input<KeyCode>>) {
    if keys.just_pressed(KeyCode::Up) {
        render_distance.0 += 1
    }
    if keys.just_pressed(KeyCode::Down) {
        render_distance.0 = i32::max(render_distance.0 - 1, 1)
    }
}

fn load_unload_visual_chunks(mut commands: Commands, player: Query<(&Player, &Transform)>, 
    already_loaded: Query<(Entity, &VisualChunk, &Handle<Mesh>)>, 
    mut meshes: ResMut<Assets<Mesh>>, 
    sand: Res<SandMaterial>,
    render_distance: Res<RenderDistance>
) {
    let player = player.get_single();
    if let Ok(player) = player {
        let middle_chunk = chunk_coordinate_of(get_horizontal(player.1.translation));
        let mut near_chunks = Vec::new();/*vec!(
            middle_chunk,
            (middle_chunk.0 + 1, middle_chunk.1),
            (middle_chunk.0 + 1, middle_chunk.1 + 1),
            (middle_chunk.0, middle_chunk.1 + 1),
            (middle_chunk.0 - 1, middle_chunk.1 + 1),
            (middle_chunk.0 - 1, middle_chunk.1),
            (middle_chunk.0 - 1, middle_chunk.1 - 1),
            (middle_chunk.0, middle_chunk.1 - 1),
            (middle_chunk.0 + 1, middle_chunk.1 - 1),
            (middle_chunk.0 + 2, middle_chunk.1),
            (middle_chunk.0 - 2, middle_chunk.1),
            (middle_chunk.0, middle_chunk.1 + 2),
            (middle_chunk.0, middle_chunk.1 - 2),
            (middle_chunk.0 + 2, middle_chunk.1 - 1),
            (middle_chunk.0 + 2, middle_chunk.1 + 1),
            (middle_chunk.0 - 2, middle_chunk.1 + 1),
            (middle_chunk.0 - 2, middle_chunk.1 - 1),
            (middle_chunk.0 - 1, middle_chunk.1 + 2),
            (middle_chunk.0 + 1, middle_chunk.1 + 2),
            (middle_chunk.0 + 1, middle_chunk.1 - 2),
            (middle_chunk.0 - 1, middle_chunk.1 - 2),
        );*/  

        near_chunks.reserve((render_distance.0^2) as usize);

        for x in middle_chunk.0-render_distance.0..middle_chunk.0+render_distance.0 {
            for y in middle_chunk.1-render_distance.0..middle_chunk.1+render_distance.0 {
                near_chunks.push((x,y))
            }
        }

        for (entity, chunk, handle) in already_loaded.iter() {
            if !near_chunks.contains(&(chunk.x,chunk.y)) {
                commands.entity(entity).despawn();
                meshes.remove(handle);
            } else {
                let index = near_chunks.iter().position(| &pos| pos == (chunk.x,chunk.y)).unwrap();
                near_chunks.remove(index);
            }
        }

        for (x, y) in near_chunks {
            /*let heights = generate_chunk(Vec2::new(
                (x * CHUNK_TILE_LENGTH as i32) as f32 - (CHUNK_TILE_LENGTH / 2) as f32,
                (y * CHUNK_TILE_LENGTH as i32) as f32 - (CHUNK_TILE_LENGTH / 2) as f32)
            );*/

            let mut chunk = commands.spawn((VisualChunk {x, y}));

            let thread_handle = thread::spawn(move|| {

                //let index = chunk.id().index();
                //let generation = chunk.id().generation();

                let heights = generate_visual_chunk(Vec2::new(
                    (x * CHUNK_TILE_LENGTH as i32) as f32 - (CHUNK_TILE_LENGTH / 2) as f32 - 1.0,
                    (y * CHUNK_TILE_LENGTH as i32) as f32 - (CHUNK_TILE_LENGTH / 2) as f32 - 1.0),
                    CHUNK_TILE_LENGTH + 3,
                );

                let mut meshBuf = HeightMeshBuffer::default();

                height_mesh(heights.as_slice(), &ChunkShape{}, [0; 2], [CHUNK_TILE_LENGTH+2; 2], &mut meshBuf);

                let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
                mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, meshBuf.positions);
                mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, meshBuf.normals);
                mesh.set_indices(Some(Indices::U32(meshBuf.indices)));

                mesh

                /*let mesh_handle = meshes.add(mesh);


                chunk.insert(PbrBundle {mesh: mesh_handle, 
                material: Handle::<StandardMaterial>::weak(sand.0.as_ref().unwrap().id()), 
                transform: Transform::from_xyz((x*CHUNK_TILE_LENGTH as i32) as f32 - (CHUNK_TILE_LENGTH / 2) as f32, -20., (y*CHUNK_TILE_LENGTH as i32) as f32 - (CHUNK_TILE_LENGTH / 2) as f32),
                ..Default::default()});

                chunk.insert(VisualChunkAnimating {});*/
                //let heightfield = Collider::heightfield(heights, CHUNK_TILE_LENGTH as usize, CHUNK_TILE_LENGTH as usize, Vec3::new(CHUNK_TILE_LENGTH as f32, 1., CHUNK_TILE_LENGTH as f32));
                
                //chunk.insert(heightfield);
            });

            chunk.insert(LoadingChunk(Some(thread_handle)));

            /*let heightfield = Collider::heightfield(heights, CHUNK_TILE_LENGTH as usize,
                * CHUNK_TILE_LENGTH as usize, Vec3::new(CHUNK_TILE_LENGTH as f32, 1.,
                    * CHUNK_TILE_LENGTH as f32));*/
            /*commands.spawn((heightfield, Chunk {x, y},
                TransformBundle::from_transform(Transform::from_xyz((x*CHUNK_TILE_LENGTH as i32) as f32, 0., (y*CHUNK_TILE_LENGTH as i32) as f32)))
            );*/
        }
    }
}

#[derive(Component)]
struct LoadingChunk(Option<JoinHandle<Mesh>>);

fn check_loading_chunks(
    mut loading: Query<(Entity, &mut LoadingChunk, &VisualChunk)>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    sand: Res<SandMaterial>,
) {
    for (entity, mut chunk, visual) in loading.iter_mut() {
        if chunk.0.as_ref().unwrap().is_finished() {
            let mesh = chunk.0.take().unwrap().join();
            if mesh.is_err() {
                continue
            }
            let mesh = mesh.unwrap();

            let mut chunk = commands.entity(entity);

            let mesh_handle = meshes.add(mesh);

            let x = visual.x;
            let y = visual.y;

            chunk.insert(PbrBundle {mesh: mesh_handle, 
            material: Handle::<StandardMaterial>::weak(sand.0.as_ref().unwrap().id()), 
            transform: Transform::from_xyz((x*CHUNK_TILE_LENGTH as i32) as f32 - (CHUNK_TILE_LENGTH / 2) as f32, -20., (y*CHUNK_TILE_LENGTH as i32) as f32 - (CHUNK_TILE_LENGTH / 2) as f32),
            ..Default::default()});

            chunk.insert(VisualChunkAnimating {});

            chunk.remove::<LoadingChunk>();
        }
    }
}

#[derive(Component)]
struct LoadingCollisionChunk(Option<JoinHandle<Collider>>);

fn check_loading_collision_chunks(
    mut loading: Query<(Entity, &mut LoadingCollisionChunk, &Chunk)>,
    mut commands: Commands,
) {
    for (entity, mut chunk, coords) in loading.iter_mut() {
        if chunk.0.as_ref().unwrap().is_finished() {
            let col = chunk.0.take().unwrap().join();
            if col.is_err() {
                continue
            }
            let col = col.unwrap();

            let mut chunk = commands.entity(entity);

            let x = coords.x;
            let y = coords.y;

            chunk.insert(col);

            chunk.remove::<LoadingCollisionChunk>();
        }
    }
}

const ANIMATE_TIME: f32 = 0.2;

fn animate_chunk_in(
    mut chunk: Query<(Entity, &VisualChunkAnimating, &mut Transform)>,
    mut commands: Commands,
    time: Res<Time>,
){
    for (entity, vAnim, mut transform) in chunk.iter_mut() {
        let y = transform.translation.y;
        if (y > -0.01) {
            transform.translation.y = 0.;
            let mut entitycommands = commands.entity(entity);
            entitycommands.remove::<VisualChunkAnimating>();
        } else {
            let dist = f32::abs(transform.translation.y);
            let travel = dist * time.delta_seconds() / ANIMATE_TIME;
            transform.translation.y += travel
        }
    }
}

const NOISE_1_SEED: u32 = 7;
const NOISE_2_SEED: u32 = 42;
const CHUNK_TILE_LENGTH: u32 = 64;
const CHUNK_TILE_SIZE: f32 = 100.;
const NOISE_AMPLITUDE: f32 = 13.;
const NOISE_DOMAIN_SCALE: f32 = 0.02;
const NOISE_MIN_SMOOTHING: f64 = 0.03;

fn generate_chunk(negativeCorner: Vec2, len: u32) -> Vec<Real> {
    let mut heights: Vec<Real> = Vec::new();

    for n in 0..(len*len) {
        let mut row = (n % len) as f32;
        let mut col = f32::floor(n as f32 / len as f32);

        if row <= 0. {row -= 0.5}
        if row >= (len - 1) as f32 {row += 0.5}
        if col <= 0. {col -= 0.5}
        if col >= (len - 1) as f32 {col += 0.5}
        

        let pos = (negativeCorner + Vec2::new(col, row))*unsafe {CONF_TERRAIN_SCALE};
    

        heights.push(terrain_height(pos));
    
        //println!("{} {} {}", n, row, col);

        //let pos: [f64; 2] = [((negativeCorner.x + row) * 0.04) as f64, ((negativeCorner.y + col) * 0.04) as f64];
        //println!("{} {}",pos[0],pos[1]);
        //let value: f64 = noise.get(pos) * NOISE_AMPLITUDE as f64;
        //println!("{}",value);
        //heights.push(value as f32);
    }

    heights
}

fn generate_visual_chunk(negativeCorner: Vec2, len: u32) -> Vec<Real> {
    let mut heights: Vec<Real> = Vec::new();

    for n in 0..(len*len) {
        let mut row = (n % len) as f32;
        let mut col = f32::floor(n as f32 / len as f32);

        if row <= 0. {row -= 0.5}
        if row >= (len - 1) as f32 {row += 0.5}
        if col <= 0. {col -= 0.5}
        if col >= (len - 1) as f32 {col += 0.5}
        

        let pos = (negativeCorner + Vec2::new(row, col))*unsafe{CONF_TERRAIN_SCALE};
    

        heights.push(terrain_height(pos));
    
        //println!("{} {} {}", n, row, col);

        //let pos: [f64; 2] = [((negativeCorner.x + row) * 0.04) as f64, ((negativeCorner.y + col) * 0.04) as f64];
        //println!("{} {}",pos[0],pos[1]);
        //let value: f64 = noise.get(pos) * NOISE_AMPLITUDE as f64;
        //println!("{}",value);
        //heights.push(value as f32);
    }

    heights
}
 

fn get_smooth_min(a: f64, b: f64) -> f64 {
    let h: f64 = f64::max(NOISE_MIN_SMOOTHING-f64::abs(a-b), 0.0)/NOISE_MIN_SMOOTHING;
    f64::min(a, b) - h*h*NOISE_MIN_SMOOTHING*0.25
}

fn terrain_height(v: Vec2) -> f32 {
    let noise1 = OpenSimplex::new(NOISE_1_SEED);
    let noise2 = OpenSimplex::new(NOISE_2_SEED);
    //let noise = Min::new(noise1, noise2);

    let pos: [f64; 2] = [v.x as f64, v.y as f64];
    
    (get_smooth_min(noise1.get(pos), noise2.get(pos)) * unsafe{CONF_TERRAIN_HEIGHT} as f64) as f32
}

const MAX_PITCH: f32 = 1.55334303;// 89 degrees

fn rotate_camera(
    mut cam: Query<(&mut CameraRotation, &mut Transform)>, 
    binds: Query<&KeybindsComponent>,
    mut motion_evr: EventReader<MouseMotion>
){
    let cam_query_opt = cam.iter_mut().next();// there will only be one
    if cam_query_opt.is_none() {return};

    let cam_query = cam_query_opt.unwrap();
    let cam = cam_query.0.into_inner();
    let transform = cam_query.1.into_inner();

    let binds_query_opt = binds.iter().next();
    if binds_query_opt.is_none() {return};

    let binds = binds_query_opt.unwrap();

    let mut total_yaw_delta = 0.;
    let mut total_pitch_delta = 0.;

    for ev in motion_evr.iter() {
        total_yaw_delta -= ev.delta.x;
        total_pitch_delta += ev.delta.y;
    }

    let initial_yaw = cam.yaw + total_yaw_delta * binds.sensitivity;
    let initial_pitch = cam.pitch - total_pitch_delta * binds.sensitivity;

    let resulting_yaw = initial_yaw % std::f32::consts::TAU;
    let resulting_pitch = initial_pitch.clamp(-MAX_PITCH, MAX_PITCH);

    cam.pitch = resulting_pitch;
    cam.yaw = resulting_yaw;
    
    let new_rotation = Quat::from_rotation_y(cam.yaw) * Quat::from_rotation_x(cam.pitch);

    transform.rotation = new_rotation;
}

fn move_camera(
    mut cam: Query<(&CameraRotation, &mut Transform)>,
    player: Query<(&Player, &GlobalTransform)>,
    time: Res<Time>
){
    for (_cr, mut cam_transform) in cam.iter_mut() {
        for (_player, player_transform) in player.iter() {
            let orig_cam_pos = cam_transform.translation;
            let player_pos = player_transform.translation();
            let target_cam_pos = player_pos + Vec3::Y * 0.625;
            let delta_time = time.delta_seconds();
            
            let resulting_cam_pos = Vec3::lerp(orig_cam_pos, target_cam_pos, f32::min(delta_time*3. + 0.1, 1.));

            cam_transform.translation = target_cam_pos;
        }
    }
}


fn get_wishdir(
    binds: Query<&KeybindsComponent>, 
    cam: Query<&CameraRotation>,
    keys: Res<Input<KeyCode>>
) -> Vec2 {
    for binds in binds.iter(){
        for cam in cam.iter(){
            let raw_vec = Vec2 {
                x: keys.pressed(binds.right) as i32 as f32 - keys.pressed(binds.left) as i32 as f32, 
                y: keys.pressed(binds.backward) as i32 as f32 - keys.pressed(binds.forward) as i32 as f32,
            }; 

            let normalized_vec = raw_vec.normalize_or_zero();

            let rotated_vec = normalized_vec.rotate(Vec2::from_angle(-cam.yaw));

            return rotated_vec;
        }
    }
    Vec2::ZERO
}

const ACCEL: f32 = 0.042;
const MAX_SPEED: f32 = 0.4;
const GRAVITY: f32 = 0.01;

fn movement(
    In(wishdir): In<Vec2>,  
    mut player: Query<(&mut Player, &Grounded, &KeybindsComponent, &mut JumpBuffer)>,
    /*mut debug: Query<(&DebugText, &mut Text)>,*/
    keys: Res<Input<KeyCode>>,
){
    for (mut player, grounded, binds, mut jump) in player.iter_mut() {
        player.vel.y -= GRAVITY;

        /*debug.get_single_mut().unwrap().1.sections[0].value = 
            format!("g: {} \nj: {}\nv: {}", grounded.0, jump.0, get_horizontal(player.vel).length());*/

        //println!("g: {}, j: {}, v: {}", grounded.0, jump.0, player.vel.length());
        if grounded.0 {
            ground_movement(&mut player, wishdir, &keys, binds, &mut jump)
        } else {
            air_movement(&mut player, wishdir)
        }
        //let horizontal_velocity = get_horizontal(velocity.linvel);
        //let current_speed = horizontal_velocity.dot(wishdir);
        //let mut add_speed = ACCEL;
        //if current_speed + add_speed > MAX_SPEED {add_speed = MAX_SPEED - current_speed}

        //velocity.linvel += horizontal_to_3d(wishdir * add_speed);

        // velocity.linvel *= 0.94;

        // println!("velocity: ({}, {}, {})",velocity.linvel.x,velocity.linvel.y,velocity.linvel.z);
    }
}

const GROUND_SPEED_THRESHOLD: f32 = MAX_SPEED*MAX_SPEED;// below this speed normal accel applies, above quake accel applies
const NORMAL_GROUND_FRICTION: f32 = 0.97;
const QUAKE_GROUND_FRICTION: f32 = 0.99;
const JUMP_FORCE: f32 = 0.125;

fn ground_movement(mut player: &mut Player, wishdir: Vec2, keys: &Res<Input<KeyCode>>, binds: &KeybindsComponent, mut jump: &mut JumpBuffer) {
    if !keys.any_pressed([binds.forward,binds.left,binds.backward,binds.right]) || get_horizontal(player.vel).length_squared() < GROUND_SPEED_THRESHOLD {
        normal_accelerate(&mut player, wishdir, NORMAL_GROUND_FRICTION, ACCEL*2.4)
    } else {
        quake_accelerate(&mut player, wishdir, QUAKE_GROUND_FRICTION, ACCEL, MAX_SPEED)
    }

    if jump.0 {
        player.vel.y = f32::max(player.vel.y + JUMP_FORCE, JUMP_FORCE);
        jump.0 = false
    }
}

const AIR_FRICTION: f32 = 1.; // none

fn air_movement(mut player: &mut Player, wishdir: Vec2) {
     quake_accelerate(&mut player, wishdir, AIR_FRICTION, ACCEL, MAX_SPEED)
}

fn normal_accelerate(mut player: &mut Player, wishdir: Vec2, friction: f32, accel: f32) {
    let mut horizontal = get_horizontal(player.vel);
    horizontal *= friction;
    horizontal += accel * wishdir;
    player.vel = horizontal_to_3d_with_y(horizontal, player.vel.y)
}

fn quake_accelerate(mut player: &mut Player, wishdir: Vec2, friction: f32, accel: f32, faulty_max_speed: f32) {
    let horizontal_velocity = get_horizontal(player.vel);
    let current_speed = horizontal_velocity.dot(wishdir);
    let mut add_speed = accel;
    if current_speed + add_speed > faulty_max_speed {add_speed = faulty_max_speed - current_speed}

    player.vel += horizontal_to_3d(wishdir * add_speed);
    
    player.vel = horizontal_to_3d_with_y(get_horizontal(player.vel) * friction, player.vel.y)
}
/*
fn ground_check(rapier: Res<RapierContext>, mut player: Query<(Entity, &Player, &mut Grounded)>) {
    for (entity, _player, mut grounded) in player.iter_mut() {
        for contact_pair in rapier.contacts_with(entity) {
            for contact in contact_pair.manifolds() {
                //println!("normal: {}", contact.normal());
                if is_floor(-contact.normal()){// normal is negative for some reason ???
                    grounded.0 = true;
                    return;
                }
            }
        }
        grounded.0 = false;
    }
}*/

const MAX_SLIDES: usize = 4;
const COLLISION_MARGIN: f32 = 0.03;

fn move_and_slide(// move the player based on velocity and slide along collisions
    rapier: Res<RapierContext>, 
    mut player: Query<(Entity, &mut Player, &mut Transform, &mut Grounded/*, &Collider*/)>,
    /*mut debug: Query<(&DebugText, &mut Text)>,*/
){
    let player = player.get_single_mut();
    if player.is_err() {return}
    let player = player.unwrap();
    let (_entity, mut player, mut transform, mut grounded/*, collider*/) = player;
    let collider = Collider::capsule_y(1., 0.8);
    /*let mut debug = debug.get_single_mut().unwrap();*/
    
    grounded.0 = false;

    let mut remaining_motion = player.vel;
    let mut vel = player.vel;

    let mut slides = 0;
    for _i in 0..MAX_SLIDES {

        //println!("{}",_i);
        let max_toi = remaining_motion.length() + COLLISION_MARGIN; 
        let cast_result = rapier.cast_shape(transform.translation, Quat::IDENTITY, remaining_motion.normalize_or_zero(), &collider, max_toi, QueryFilter::new());
        
        if cast_result.is_none() {
            transform.translation += remaining_motion;
            break
        }

        let (_entity, toi) = cast_result.unwrap();
        
        if is_floor(toi.normal1) {grounded.0 = true}

        //let applied_toi = (toi.toi - (-toi.normal1.dot(remaining_motion) * COLLISION_MARGIN)).max(0.);

        let mut applied_motion = remaining_motion.normalize_or_zero() * (toi.toi);
        applied_motion += toi.normal1 * COLLISION_MARGIN;
        remaining_motion = slide(remaining_motion, toi.normal1).normalize_or_zero() * (max_toi - toi.toi);
        
        //if applied_motion.length() < COLLISION_MARGIN {applied_motion = Vec3::ZERO}

        transform.translation += applied_motion;

        //remaining_motion = slide(remaining_motion - applied_motion, toi.normal1).normalize_or_zero() * (max_toi - toi.toi);
        vel = slide(vel,toi.normal1);
        slides += 1;
    }
    
    let velocity_delta = f32::abs((player.vel-vel).length()); 

    /*if velocity_delta > 0.05 {debug.1.sections[0].value = format!("velocity delta from collisions: {}\nslides: {}", velocity_delta, slides)}*/

    player.vel = vel;
}

fn do_jump_buffering(
    mut query: Query<(&mut JumpBuffer, &KeybindsComponent)>,
    keys: Res<Input<KeyCode>>,
){
    for (mut buf, binds) in query.iter_mut() {
        if keys.just_pressed(binds.jump) {buf.0 = true}
        if keys.just_released(binds.jump) {buf.0 = false}
    }
}

const FLOOR: Vec3 = Vec3 {x:0., y:1., z:0.};

const FLOOR_MIN: f32 = 0.8;

fn is_floor(v: Vec3) -> bool {
    v.dot(FLOOR) > FLOOR_MIN
}

fn get_horizontal(v: Vec3) -> Vec2 {
    Vec2 {x: v.x, y: v.z}
}

fn horizontal_to_3d(v: Vec2) -> Vec3 {
    Vec3 {x: v.x, y: 0., z: v.y}
}

fn horizontal_to_3d_with_y(v: Vec2, y: f32) -> Vec3 {
    horizontal_to_3d(v) + Vec3::Y * y
}

fn cursor_locking(
    keys: Res<Input<KeyCode>>,
    buttons: Res<Input<MouseButton>>,
    mut window: Query<&mut Window>,
){
    let mut window = window.single_mut();

    if buttons.just_pressed(MouseButton::Left){
        window.cursor.visible = false;
        window.cursor.grab_mode = CursorGrabMode::Locked;
    }

    if keys.just_pressed(KeyCode::Escape){
        window.cursor.visible = true;
        window.cursor.grab_mode = CursorGrabMode::None;
    }
}

fn slide(vec: Vec3, norm: Vec3) -> Vec3 {// remove projection of vec onto norm from vec, remaining vec is parallel to the plane defined by norm.
    vec - norm * norm.dot(vec)
}
/*
#[derive(Component)]
struct DebugText;

fn add_debug_text(assets: Res<AssetServer>, mut commands: Commands){
    let font = assets.load("font.ttf");
    commands.spawn(TextBundle::from_section(
        "debug_text",
        TextStyle { font: font.clone(), font_size: 25., color: Color::WHITE })
        .with_text_alignment(TextAlignment::Left)
    ).insert(DebugText{});
}*/
