use anyhow::Result;
use std::{fs::File, path::Path};
use tiny_http::Method;

fn main() -> Result<()> {
    let server = tiny_http::Server::http("0.0.0.0:8888").unwrap();

    loop {
        let request = match server.recv() {
            Ok(rq) => rq,
            Err(err) => {
                println!("Request fail brah: {}", err);
                break Ok(());
            }
        };

        match request.method() {
            Method::Get => match request.url() {
                // home html pages
                "/" => {
                    let response =
                        tiny_http::Response::from_file(File::open(&Path::new("ui/homepage.html"))?);
                    request.respond(response)?;
                }
                "/stylesheet.css" => {
                    let response = tiny_http::Response::from_file(File::open(&Path::new(
                        "ui/stylesheet.css",
                    ))?);
                    request.respond(response)?;
                }
                "/landing" => {
                    let response = tiny_http::Response::from_file(File::open(&Path::new(
                        "ui/indoctornate.html",
                    ))?);
                    request.respond(response)?;
                }

                // image fetches
                "/swordperator.gif" => {
                    let response = tiny_http::Response::from_file(File::open(&Path::new(
                        "ui/images/swordperator.gif",
                    ))?);
                    request.respond(response)?;
                }
                "/starry.png" => {
                    let response = tiny_http::Response::from_file(File::open(&Path::new(
                        "ui/images/starry.png",
                    ))?);
                    request.respond(response)?;
                }
                "/BrownDude.png" => {
                    let response = tiny_http::Response::from_file(File::open(&Path::new(
                        "ui/images/BrownDude.png",
                    ))?);
                    request.respond(response)?;
                }
                "/BrickWallWaterfall.jpg" => {
                    let response = tiny_http::Response::from_file(File::open(&Path::new(
                        "ui/images/BrickWallWaterfall.jpg",
                    ))?);
                    request.respond(response)?;
                }
                "/RockAndStone.jpg" => {
                    let response = tiny_http::Response::from_file(File::open(&Path::new(
                        "ui/images/RockAndStone.jpg",
                    ))?);
                    request.respond(response)?;
                }
                "/Jumpscare.png" => {
                    let response = tiny_http::Response::from_file(File::open(&Path::new(
                        "ui/images/Jumpscare.png",
                    ))?);
                    request.respond(response)?;
                }
                "/bust_guy_vibed.png" => {
                    let response = tiny_http::Response::from_file(File::open(&Path::new(
                        "ui/images/bust_guy_vibed.png",
                    ))?);
                    request.respond(response)?;
                }
                "/Hallway.png" => {
                    let response = tiny_http::Response::from_file(File::open(&Path::new(
                        "ui/images/Hallway.png",
                    ))?);
                    request.respond(response)?;
                }
                "/Topper.png" => {
                    let response = tiny_http::Response::from_file(File::open(&Path::new(
                        "ui/images/Topper.png",
                    ))?);
                    request.respond(response)?;
                }
                "/OrbPedestalUnlit.png" => {
                    let response = tiny_http::Response::from_file(File::open(&Path::new(
                        "ui/images/OrbPedestalUnlit.png",
                    ))?);
                    request.respond(response)?;
                }
                "/OrbPedestalLit.png" => {
                    let response = tiny_http::Response::from_file(File::open(&Path::new(
                        "ui/images/OrbPedestalLit.png",
                    ))?);
                    request.respond(response)?;
                }
                _ => {
                    let response = tiny_http::Response::from_string("Error 404: page not found");
                    request.respond(response)?;
                }
            },
            Method::Head => todo!(),
            Method::Post => todo!(),
            Method::Put => todo!(),
            Method::Delete => todo!(),
            Method::Connect => todo!(),
            Method::Options => todo!(),
            Method::Trace => todo!(),
            Method::Patch => todo!(),
            Method::NonStandard(_ascii_string) => todo!(),
        }
    }
}
