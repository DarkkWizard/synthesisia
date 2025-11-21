use anyhow::Result;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample, I24,
};
use std::{
    f32,
    fs::{self, File},
    path::Path,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

fn main() -> anyhow::Result<()> {
    let stream = stream_setup()?;
    stream.play()?;
    let server = tiny_http::Server::http("0.0.0.0:8888").expect("failed to create server");
    let server_is_running = true;
    while server_is_running {
        for request in server.incoming_requests() {
            let url = request.url();

            let path = if url == "/" {
                "ui/homepage.html".to_string()
            } else {
                format!("ui{}", url)
            };

            match fs::read(&path) {
                Ok(_) => {
                    let response = tiny_http::Response::from_file(File::open(Path::new(&path))?);
                    request.respond(response)?;
                }
                Err(_) => {
                    let response = tiny_http::Response::from_string("Error 404: that don't exist");
                    request.respond(response)?;
                }
            }
        }
    }
    Ok(())
}

const A: f32 = 27.50;
const ASHARP: f32 = 29.14;
const B: f32 = 30.87;
const C: f32 = 16.35;
const CSHARP: f32 = 17.32;
const D: f32 = 18.35;
const DSHARP: f32 = 19.45;
const E: f32 = 20.60;
const F: f32 = 21.83;
const FSHARP: f32 = 23.12;
const G: f32 = 24.50;
const GSHARP: f32 = 25.96;

#[derive(Clone, Debug)]
pub struct Synth {
    pub voices: Vec<Voice>,
    pub sample_rate: f32,
    adsr: AdsrEnvelope,
}

impl Synth {
    pub fn new(sample_rate: f32, num_voices: usize, adsr: AdsrEnvelope) -> Self {
        let mut voices = Vec::new();

        for _ in 0..num_voices {
            voices.push(Voice {
                oscillator: Oscillator {
                    sample_rate,
                    waveform: Waveform::Sine,
                    current_sample_index: 0.,
                    frequency_hz: 0.,
                },
                is_active: false,
                note: Notes::C(4),
                age: 0,
                adsr: AdsrEnvelope::new_defaults(),
            })
        }

        Self {
            voices,
            sample_rate,
            adsr,
        }
    }

    pub fn press_it_pops(&mut self, note: Notes) {
        // Find an inactive voice or steal the oldest
        if let Some(_) = self.voices.iter().find(|v| v.note == note) {
            return;
        }

        if let Some(voice) = self.voices.iter_mut().find(|v| !v.is_active) {
            voice.is_active = true;
            voice.note = note;
            voice.oscillator.frequency_hz = note_to_num(note);
            voice.oscillator.current_sample_index = 0.0; // Reset phase
            voice.age = 0;
            return;
        }

        if let Some(oldest_voice) = self.voices.iter_mut().max_by_key(|v| v.age) {
            oldest_voice.is_active = true;
            oldest_voice.note = note;
            oldest_voice.oscillator.frequency_hz = note_to_num(note);
            oldest_voice.oscillator.current_sample_index = 0.0; // Reset phase
            oldest_voice.age = 0;
        }
    }

    // stop playing that note
    pub fn careful_now(&mut self, note: Notes) {
        for voice in self.voices.iter_mut() {
            if voice.is_active && voice.note == note {
                voice.is_active = false;
            }
        }
    }

    pub fn tick(&mut self) -> f32 {
        let mut output = 0.0;
        let mut active_count = 0;

        for voice in self.voices.iter_mut() {
            if voice.is_active {
                output += voice.oscillator.tick();
                active_count += 1;
                voice.age += 1;
            }
        }

        let out = if active_count > 0 {
            (output / active_count as f32).clamp(-0.9, 0.9)
        } else {
            0.0
        };
        out
    }

    fn major_triad(&mut self, root: Notes) {
        self.press_it_pops(root);
        self.press_it_pops(root + 4);
        self.press_it_pops(root + 7);
    }

    fn minor_triad(&mut self, root: Notes) {
        self.press_it_pops(root);
        self.press_it_pops(root + 3);
        self.press_it_pops(root + 7);
    }

    fn clear(&mut self) {
        for voice in self.voices.iter_mut() {
            voice.is_active = false;
            voice.age = 0;
            voice.oscillator.current_sample_index = 0.;
            voice.oscillator.frequency_hz = 0.;
        }
    }

    pub fn change_envelope(&mut self) {}

    fn propogate_envelope_change(&mut self) {
        for mut voice in self.voices.clone() {
            voice.adsf = self.adsr.clone();
        }
    }

    fn change_adsr_attack(&mut self, increment: Duration) {
        self.adsr.increment_attack(increment);
    }

    fn change_adsr_decay(&mut self, increment: Duration) {
        self.adsr.increment_decay(increment);
    }

    fn change_adsr_sustain(&mut self, increment: f32) {
        self.adsr.increment_sustain_level(increment);
    }

    fn change_adsr_release(&mut self, increment: Duration) {
        self.adsr.increment_release(increment);
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Waveform {
    Sine,
    Square,
    Saw,
    Triangle,
}

#[derive(Copy, Clone, Debug)]
pub struct AdsrEnvelope {
    attack: Duration,
    decay: Duration,
    sustain: f32,
    release: Duration,
}

impl AdsrEnvelope {
    pub fn new(attack: Duration, decay: Duration, sustain: f32, release: Duration) -> Self {
        Self {
            attack,
            decay,
            sustain,
            release,
        }
    }

    pub fn new_defaults() -> Self {
        Self {
            attack: Duration::from_millis(100),
            decay: Duration::from_millis(100),
            sustain: 1.,
            release: Duration::from_millis(300),
        }
    }

    pub fn increment_attack(&mut self, increment: Duration) {
        self.attack += increment;
    }

    pub fn increment_decay(&mut self, increment: Duration) {
        self.decay += increment;
    }

    pub fn increment_sustain_level(&mut self, increment: f32) {
        self.sustain += increment;
    }

    pub fn increment_release(&mut self, increment: Duration) {
        self.release += increment;
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Oscillator {
    pub sample_rate: f32,
    pub waveform: Waveform,
    pub current_sample_index: f32,
    pub frequency_hz: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct Voice {
    pub oscillator: Oscillator,
    pub is_active: bool,
    pub note: Notes,
    pub age: usize,
    pub adsf: AdsrEnvelope,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Notes {
    A(u32),
    Asharp(u32),
    B(u32),
    C(u32),
    Csharp(u32),
    D(u32),
    Dsharp(u32),
    E(u32),
    F(u32),
    Fsharp(u32),
    G(u32),
    Gsharp(u32),
}

impl Notes {
    fn to_semitone(&self) -> u32 {
        match self {
            Notes::C(_) => 0,
            Notes::Csharp(_) => 1,
            Notes::D(_) => 2,
            Notes::Dsharp(_) => 3,
            Notes::E(_) => 4,
            Notes::F(_) => 5,
            Notes::Fsharp(_) => 6,
            Notes::G(_) => 7,
            Notes::Gsharp(_) => 8,
            Notes::A(_) => 9,
            Notes::Asharp(_) => 10,
            Notes::B(_) => 11,
        }
    }

    fn octave(&self) -> u32 {
        match self {
            Notes::C(o)
            | Notes::Csharp(o)
            | Notes::D(o)
            | Notes::Dsharp(o)
            | Notes::E(o)
            | Notes::F(o)
            | Notes::Fsharp(o)
            | Notes::G(o)
            | Notes::Gsharp(o)
            | Notes::A(o)
            | Notes::Asharp(o)
            | Notes::B(o) => *o,
        }
    }

    fn from_semitone_and_octave(semitone: u32, octave: u32) -> Self {
        match semitone % 12 {
            0 => Notes::C(octave),
            1 => Notes::Csharp(octave),
            2 => Notes::D(octave),
            3 => Notes::Dsharp(octave),
            4 => Notes::E(octave),
            5 => Notes::F(octave),
            6 => Notes::Fsharp(octave),
            7 => Notes::G(octave),
            8 => Notes::Gsharp(octave),
            9 => Notes::A(octave),
            10 => Notes::Asharp(octave),
            11 => Notes::B(octave),
            _ => unreachable!(),
        }
    }
}

impl std::ops::Add<u32> for Notes {
    type Output = Notes;

    fn add(self, semitones: u32) -> Self::Output {
        let current_semitone = self.to_semitone();
        let current_octave = self.octave();

        let total_semitones = current_semitone + semitones;
        let new_semitone = total_semitones % 12;
        let octave_increase = total_semitones / 12;
        let new_octave = current_octave + octave_increase;

        Notes::from_semitone_and_octave(new_semitone, new_octave)
    }
}

pub fn note_to_num(note: Notes) -> f32 {
    match note {
        Notes::A(x) => A * 2_u32.pow(x) as f32,
        Notes::Asharp(x) => ASHARP * 2_u32.pow(x) as f32,
        Notes::B(x) => B * 2_u32.pow(x) as f32,
        Notes::C(x) => C * 2_u32.pow(x) as f32,
        Notes::Csharp(x) => CSHARP * 2_u32.pow(x) as f32,
        Notes::D(x) => D * 2_u32.pow(x) as f32,
        Notes::Dsharp(x) => DSHARP * 2_u32.pow(x) as f32,
        Notes::E(x) => E * 2_u32.pow(x) as f32,
        Notes::F(x) => F * 2_u32.pow(x) as f32,
        Notes::Fsharp(x) => FSHARP * 2_u32.pow(x) as f32,
        Notes::G(x) => G * 2_u32.pow(x) as f32,
        Notes::Gsharp(x) => GSHARP * 2_u32.pow(x) as f32,
    }
}

impl Oscillator {
    fn advance_sample(&mut self) {
        let s = self.current_sample_index;
        self.current_sample_index = if s > 340282350000000000000000000000000000. {
            0.
        } else {
            s + 1.
        }
    }

    fn set_waveform(&mut self, waveform: Waveform) {
        self.waveform = waveform;
    }

    fn calculate_sine_output_from_freq(&self, freq: f32) -> f32 {
        let two_pi = 2.0 * std::f32::consts::PI;
        (self.current_sample_index * freq * two_pi / self.sample_rate).sin()
    }

    fn is_multiple_of_freq_above_nyquist(&self, multiple: f32) -> bool {
        self.frequency_hz * multiple > self.sample_rate / 2.0
    }

    fn sine_wave(&mut self) -> f32 {
        self.advance_sample();
        self.calculate_sine_output_from_freq(self.frequency_hz)
    }

    fn generative_waveform(&mut self, harmonic_index_increment: i32, gain_exponent: f32) -> f32 {
        self.advance_sample();
        let mut output = 0.0;
        let mut i = 1;
        while !self.is_multiple_of_freq_above_nyquist(i as f32) {
            let gain = 1.0 / (i as f32).powf(gain_exponent);
            output += gain * self.calculate_sine_output_from_freq(self.frequency_hz * i as f32);
            i += harmonic_index_increment;
        }
        output
    }

    fn square_wave(&mut self) -> f32 {
        self.generative_waveform(2, 1.0)
    }

    fn saw_wave(&mut self) -> f32 {
        self.generative_waveform(1, 1.0)
    }

    fn triangle_wave(&mut self) -> f32 {
        self.generative_waveform(2, 2.0)
    }

    fn tick(&mut self) -> f32 {
        match self.waveform {
            Waveform::Sine => self.sine_wave(),
            Waveform::Square => self.square_wave(),
            Waveform::Saw => self.saw_wave(),
            Waveform::Triangle => self.triangle_wave(),
        }
    }
}

pub fn stream_setup() -> Result<cpal::Stream, anyhow::Error>
where
{
    let (_host, device, config) = host_device_setup()?;

    match config.sample_format() {
        cpal::SampleFormat::I8 => make_stream::<i8>(&device, &config.into()),
        cpal::SampleFormat::I16 => make_stream::<i16>(&device, &config.into()),
        cpal::SampleFormat::I24 => make_stream::<I24>(&device, &config.into()),
        cpal::SampleFormat::I32 => make_stream::<i32>(&device, &config.into()),
        cpal::SampleFormat::I64 => make_stream::<i64>(&device, &config.into()),
        cpal::SampleFormat::U8 => make_stream::<u8>(&device, &config.into()),
        cpal::SampleFormat::U16 => make_stream::<u16>(&device, &config.into()),
        cpal::SampleFormat::U32 => make_stream::<u32>(&device, &config.into()),
        cpal::SampleFormat::U64 => make_stream::<u64>(&device, &config.into()),
        cpal::SampleFormat::F32 => make_stream::<f32>(&device, &config.into()),
        cpal::SampleFormat::F64 => make_stream::<f64>(&device, &config.into()),
        sample_format => Err(anyhow::Error::msg(format!(
            "Unsupported sample format '{sample_format}'"
        ))),
    }
}

pub fn host_device_setup(
) -> Result<(cpal::Host, cpal::Device, cpal::SupportedStreamConfig), anyhow::Error> {
    let host = cpal::default_host();

    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::Error::msg("Default output device is not available"))?;
    println!("Output device : {}", device.name()?);

    let config = device.default_output_config()?;
    println!("Default output config : {config:?}");

    Ok((host, device, config))
}

pub fn make_stream<T>(device: &cpal::Device, config: &cpal::StreamConfig) -> Result<cpal::Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let synth = Arc::new(Mutex::new(Synth::new(
        config.sample_rate.0 as f32,
        8,
        AdsrEnvelope::new_defaults(),
    ))); // 8-voice polyphony
    let num_channels = config.channels as usize;
    let err_fn = |err| eprintln!("Error building output sound stream: {err}");

    let synth_clone = Arc::clone(&synth);

    thread::spawn(move || {
        {
            let mut s = synth_clone.lock().unwrap();

            for v in s.voices.iter_mut() {
                v.oscillator.set_waveform(Waveform::Sine);
            }
            s.minor_triad(Notes::A(4));
        }

        thread::sleep(Duration::from_millis(2000));

        {
            let mut s = synth_clone.lock().unwrap();
            s.clear();
            s.minor_triad(Notes::Fsharp(4));
            s.press_it_pops(Notes::F(4));
        }
        thread::sleep(Duration::from_millis(2000));

        {
            let mut s = synth_clone.lock().unwrap();
            s.clear();
            s.minor_triad(Notes::Gsharp(4));
            s.press_it_pops(Notes::Fsharp(4));
        }
    });

    let synth_render = Arc::clone(&synth);
    let stream = device.build_output_stream(
        config,
        move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
            for frame in output.chunks_mut(num_channels) {
                let value: T = T::from_sample(synth_render.lock().unwrap().tick());
                for sample in frame.iter_mut() {
                    *sample = value;
                }
            }
        },
        err_fn,
        None,
    )?;

    Ok(stream)
}
