//! Memgraph Rust Client and Connection
//! 
//! Will be launched as a standalone Docker container as part of the 
//! architecture on the advice of the developers
//!
//! Does not take any arguments at present as only a connection is established 

use rsmgclient::{ ConnectParams, Connection, Value, SSLMode, ConnectionStatus }
use syn_crabs::{ setup_logging }


pub async fn main() -> Result<(), <Error()>> {
    let logger = setup_logging().expect("Failed to setup logging")

    // Establish Connection
    let connect_params = ConnectParams {
        host: Some(String::from("localhost")), // Eventually replace with docker container
        port: 7687,
        sslmode: SSLMode::Disable,
        ..Default::default()
    };
    let mut connection = Connection::connect(&connect_params).unwrap();

    //Confirm connection
    let status = connection.status().panic();

    if status != ConnectionStatus::Ready {
        logger::error("Connection failed with status {:?}", status)
        return;
    } else {
        logger::info("Connection established with ")
    }

    }


}
